"""Knowledge tool adapters over the common linked Markdown source owner."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import List

from ouroboros import knowledge as knowledge_store
from ouroboros.knowledge import INDEX_FILE as INDEX_FILE, OVERVIEW_TOPIC as OVERVIEW_TOPIC
from ouroboros.knowledge import sanitize_topic as _sanitize_topic
from ouroboros.tools.arg_feedback import ignored_argument_note
from ouroboros.tools.registry import ToolEntry, ToolContext
from ouroboros.tools.tool_result import ToolResult, _MAX_META_BYTES, _publish_tool_result, completed_local_read
from ouroboros.utils import append_jsonl, utc_now_iso

KNOWLEDGE_DIR = "memory/knowledge"
BACKLOG_TOPIC = "improvement-backlog"
PATTERNS_TOPIC = "patterns"
# Reserved shared topics with exactly one home. Whichever room asks for them,
# they resolve to the global shelf: the backlog is the P7 SSOT, the overview is
# the shared orientation every context loads, and the Pattern Register is
# cross-project cognition whose only writer (post-task reflection) and only
# readers (context, deep self-review, the headless copy) all use the canonical
# drive. A project copy of any of them would be a second source of truth nobody
# reads.
GLOBAL_ONLY_TOPICS = knowledge_store.ALWAYS_ACTIVE_TOPICS
# Existing consolidator and Pattern Register imports share this exact lock.
_knowledge_write_lock = knowledge_store.knowledge_write_lock


def _backlog_root(ctx: ToolContext) -> Path:
    return Path(str(getattr(ctx, "budget_drive_root", "") or ctx.drive_root))


def _address(ctx: ToolContext, topic: str, scope: str = "") -> knowledge_store.KnowledgeAddress:
    if not isinstance(scope, str):
        raise ValueError("scope must be global or project:<exact project id>")
    sanitized = _sanitize_topic(topic)
    if sanitized in GLOBAL_ONLY_TOPICS:
        return knowledge_store.resolve_knowledge_address(_backlog_root(ctx), sanitized, "global")
    project = str(getattr(ctx, "project_id", "") or "")
    root = _backlog_root(ctx)
    if not getattr(ctx, "budget_drive_root", "") and (project or scope.startswith("project:")):
        from ouroboros.config import DATA_DIR
        root = Path(DATA_DIR)
    return knowledge_store.resolve_knowledge_address(root, sanitized, scope, project)


def _source_view(note: knowledge_store.KnowledgeNote, start_char: int | None = None,
                 end_char: int | None = None) -> tuple[str, dict]:
    """One exact character range of the note; a bound that asks for nothing is not an error.

    A missing bound is the note's own edge, ``(0, 0)`` selects nothing and reads the
    whole note, and an ``end_char`` past the end is lowered to it (like ``read_file``).
    Each is said in the header; the returned range is always the range delivered.
    """
    ref = note.source_ref()
    ref["state"] = note.state
    text = note.text
    total = len(text)
    if any(bound is not None and type(bound) is not int for bound in (start_char, end_char)):
        raise ValueError(f"Knowledge source range start_char={start_char!r}, end_char={end_char!r} "
                         f"must be integers; this note has complete_chars={total}")
    start, end = (0 if start_char is None else start_char), (total if end_char is None else end_char)
    notes = []
    if (start, end) == (0, 0) and total:
        notes.append(ignored_argument_note("end_char", 0, "a 0..0 range selects nothing; returned the whole note"))
        end = total
    elif end > total:
        notes.append(ignored_argument_note("end_char", end, f"the note ends at {total}; returned {start}..{total}"))
        end = total
    if not 0 <= start <= end:
        raise ValueError(f"Knowledge source range start_char={start_char!r}, end_char={end_char!r} is outside "
                         f"this note: complete_chars={total}; use 0 <= start_char <= end_char")
    ref.update(start_char=start, end_char=end, complete_chars=total)
    body = text[start:end]
    header = "[Knowledge source] " + json.dumps(ref, ensure_ascii=False, sort_keys=True) + "\n"
    header += "".join(f"[Range note] {item}\n" for item in notes)
    if note.parse_error:
        header += "[Metadata unavailable; the complete original Markdown follows.]\n"
    if note.archive_error:
        header += f"[Archive metadata invalid; kept active: {note.archive_error}.]\n"
    header += "\n"
    return header + body, {
        "knowledge_source": ref, "knowledge_body_start": len(header),
        "knowledge_body_chars": len(body), "knowledge_source_complete": start == 0 and end == total,
    }


def _knowledge_read(ctx: ToolContext, topic: str, scope: str = "",
                    start_char: int | None = None, end_char: int | None = None) -> str:
    try:
        sanitized = _sanitize_topic(topic)
        address = _address(ctx, sanitized, scope)
        note = knowledge_store.read_knowledge_note(address)
        text, meta = _source_view(note, start_char, end_char)
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


def _record_backlog_history(backlog_file: Path, topic: str, mode: str, task_id: str) -> None:
    """Audit a backlog write to the GLOBAL knowledge history (C10.1), mirroring the
    generic knowledge-history schema so the backlog's audit trail lives with the
    other global knowledge, not in a project store. Best-effort; never raises."""
    try:
        history_path = backlog_file.parent.parent / "knowledge_history.jsonl"
        new_content = backlog_file.read_text(encoding="utf-8") if backlog_file.exists() else ""
        # CPL4-C17: the sidecar-locked append seam every other journal uses —
        # a raw open("a") could tear a line under concurrent writers.
        append_jsonl(history_path, {
            "ts": utc_now_iso(),
            "task_id": task_id,
            "topic": topic,
            "mode": f"{mode}->merge",
            "new_sha256": hashlib.sha256(new_content.encode("utf-8")).hexdigest() if new_content else "",
        })
    except Exception:
        pass



def _bound_delta_meta(meta: dict) -> None:
    """Keep the tool receipt inside its existing metadata limit; history holds the full delta."""
    delta = meta.get("knowledge_delta") or {}
    headings = delta.get("removed_headings")
    if not isinstance(headings, list) or len(json.dumps(meta, ensure_ascii=True, sort_keys=True,
                                                        separators=(",", ":")).encode("utf-8")) <= _MAX_META_BYTES:
        return
    meta["knowledge_delta"] = {**delta, "removed_headings": [],
                               "removed_headings_count": len(headings), "removed_headings_omitted": True,
                               "removed_headings_sha256": hashlib.sha256(
                                   json.dumps(headings, ensure_ascii=False).encode("utf-8")).hexdigest()}


def _capture_previous(ctx: ToolContext, note: knowledge_store.KnowledgeNote) -> dict:
    """Expose exact pre-transition bytes through existing actor source custody.

    Project history is not readable via runtime_data file tools. The ordinary
    artifact reader works on both shelves. Publish to the actor's drive and the
    canonical drive used by post-task reflection before terminal copyback. The
    writer calls this under its source lock, before publication; durable shelf
    history remains authoritative.
    """
    from ouroboros.artifacts import persist_exact_text_source

    # read_file's text ABI normalizes newlines. JSON escapes retain the exact
    # Markdown (CRLF included) through that reader; decode old_content as UTF-8.
    source = json.dumps({"old_content": note.text, "revision": note.revision}, ensure_ascii=True)
    for root in dict.fromkeys((Path(ctx.drive_root), _backlog_root(ctx))):
        _text, ref, issue = persist_exact_text_source(
            root, ctx.task_id, source_id="knowledge-previous", text=source)
        if issue:
            raise OSError(issue["reason"])
    return {**ref, "format": "json", "field": "old_content"}


def _knowledge_write(
    ctx: ToolContext, topic: str, content: str | None = None, mode: str = "overwrite",
    scope: str = "", expected_revision: str | None = None, old_str: str | None = None,
    summary: str | None = None, reason: str = "",
) -> str:
    try:
        sanitized = _sanitize_topic(topic)
        lifecycle = mode in ("archive", "restore")
        if mode not in ("overwrite", "append", "edit", "archive", "restore") or not isinstance(content, (str, type(None))):
            raise ValueError("content must be Markdown; mode must be overwrite, append, edit, archive or restore")
        if mode == "archive" and sanitized in GLOBAL_ONLY_TOPICS:
            raise ValueError("overview, patterns and improvement-backlog must stay active")
        if not isinstance(reason, str) or (mode == "archive" and not reason.strip()) or (reason and mode != "archive"):
            raise ValueError("reason is required for archive and used only with archive")
        summary = None if summary == "" else summary  # an empty summary asks for nothing
        if summary is not None and (not isinstance(summary, str) or not summary.strip()):
            raise ValueError(f"summary={summary!r} is not summary text; pass the revised summary, "
                             "or omit summary to keep the current one")
        if mode != "edit" and old_str is not None:
            raise ValueError("old_str is used only with mode=edit")
        if mode != "edit" and summary is not None:
            raise ValueError(f"summary is used only with mode=edit, not mode={mode}; revise it with "
                             "mode=edit, or in the frontmatter of an overwrite's content")
        if mode != "edit" and not lifecycle and content is None:
            raise ValueError(f"mode={mode} requires content")
        if lifecycle and content is None:
            content = ""
        if mode == "edit" and summary is not None and old_str in (None, ""):  # "" asks for nothing
            if content:
                raise ValueError(f"content ({len(content)} chars) has no old_str to replace; pass old_str "
                                 "for a body edit, or omit content to revise only the summary")
            content, old_str = "", None
        elif mode == "edit" and (not isinstance(old_str, str) or not old_str):
            raise ValueError("mode=edit requires a non-empty old_str, or a summary")
        elif mode == "edit" and content is None:
            raise ValueError("mode=edit with old_str requires content, its replacement "
                             "(empty text deletes the old_str span)")
        if sanitized == BACKLOG_TOPIC and not lifecycle:
            if mode == "edit":
                raise ValueError("The improvement backlog has its own merge writer; edit is not supported")
            from ouroboros.improvement_backlog import backlog_path, merge_backlog_text
            root = _backlog_root(ctx)
            merged = merge_backlog_text(root, content)
            if merged < 0:
                raise ValueError("The improvement-backlog requires parseable ### ibl-<id> blocks with - summary: lines; the global backlog was preserved")
            _record_backlog_history(backlog_path(root), sanitized, mode, str(getattr(ctx, "task_id", "") or ""))
            return f"✅ Knowledge '{sanitized}' merged into the global backlog ({merged} item(s))."
        # The turn is the writer; the route stamp is the route that ANSWERED the
        # loop's last round (provider + resolved model, account when Claudexor
        # served it), recorded by the loop, otherwise honestly unknown. The host
        # signs which focus wrote it (``focus_signature``), never the writer's own claim.
        result = knowledge_store.write_knowledge_note(
            _address(ctx, sanitized, scope), content, mode, expected_revision,
            str(getattr(ctx, "task_id", "") or ""), old_str, writer="turn",
            route=(getattr(ctx, "_accumulated_usage", None) or {}).get("_observed_route") or None,
            summary=summary, focus=knowledge_store.focus_signature(ctx)["focus"], reason=reason,
            capture_previous=(lambda note: _capture_previous(ctx, note)) if lifecycle else None,
        )
    except ValueError as exc:
        return _publish_tool_result(ctx, ToolResult(
            status="error", code="TOOL_ARG_ERROR", text=f"⚠️ TOOL_ARG_ERROR: {exc}"))
    except (OSError, UnicodeDecodeError) as exc:
        return _publish_tool_result(ctx, ToolResult(
            status="error", code="TOOL_REPORTED_FAILURE", text=f"⚠️ TOOL_ERROR: Knowledge write failed: {type(exc).__name__}"))
    meta = {"knowledge_write_reason": result.reason}
    if result.history_ref is not None:
        meta["knowledge_previous_source"] = result.history_ref
    if result.delta is not None:
        meta["knowledge_delta"] = result.delta
    if result.current is not None:
        meta["knowledge_source"] = result.current.source_ref()
        meta["knowledge_state"] = result.current.state
    if result.ok:
        _bound_delta_meta(meta)
        return _publish_tool_result(ctx, ToolResult(
            status="ok", code="OK",
            text=f"✅ Knowledge '{sanitized}' {result.reason} ({mode}).\n" + json.dumps(meta, ensure_ascii=False, sort_keys=True),
            meta=meta))
    text = f"⚠️ TOOL_ERROR: Knowledge write not completed: {result.reason}."
    if result.reason in ("revision_required", "revision_conflict"):
        text += " Read the current source and pass its revision as expected_revision when replacing it."
    if result.current is not None:
        view, current_meta = _source_view(result.current)
        meta.update(current_meta)
        # The complete current source is useful immediately, without a hidden
        # retry or another model deciding whether the stale write was safe.
        text += "\n\n" + view
        meta["knowledge_body_start"] = len(text) - len(result.current.text)
    _bound_delta_meta(meta)
    return _publish_tool_result(ctx, ToolResult(
        status="error", code="TOOL_REPORTED_FAILURE", text=text, meta=meta))


@completed_local_read
def _knowledge_list(ctx: ToolContext, scope: str = "", view: str = "active") -> str:
    try:
        address = _address(ctx, "topic", scope)
        return knowledge_store.knowledge_index_view(address, view)
    except ValueError as exc:
        return _publish_tool_result(ctx, ToolResult(
            status="error", code="TOOL_ARG_ERROR", text=f"⚠️ TOOL_ARG_ERROR: {exc}"))


def get_tools() -> List[ToolEntry]:
    # The chronicle tools ride this module's export: the packaged build's frozen
    # module list already names it, and tools/chronicle.py has no get_tools of its own.
    from ouroboros.tools.chronicle import chronicle_tools

    topic = {"type": "string", "description": "Shelf-relative topic path without .md; nested paths and Unicode names are supported; no scope prefixes (global/, project/)."}
    scope = {"type": "string", "description": "global or project:<exact project id>. Omitted uses this task's project shelf, otherwise global. Global knowledge remains explicitly reachable from a project. Understanding of people and relationships, and anything that should outlive the project, belongs in global. Reserved topics (improvement-backlog, overview, patterns) always resolve to global."}
    return [
        ToolEntry("knowledge_read", {
            "name": "knowledge_read",
            "description": "Read the current complete knowledge note, active or archived, with its state, canonical address and exact source revision. Paths and relative Markdown links survive archival. Read current understanding before revising it.",
            "parameters": {"type": "object", "properties": {"topic": topic, "scope": scope,
                "start_char": {"type": "integer", "description": "Optional half-open character range start in the exact complete note (omitted = 0). Use ranges to read a large source in parts."},
                "end_char": {"type": "integer", "description": "Exclusive range end (omitted = the end; a value past the end is lowered to it); complete_chars and revision are returned with every view. Omit both bounds, or pass 0 and 0, for the full note."}}, "required": ["topic"]},
        }, _knowledge_read),
        ToolEntry("knowledge_write", {
            "name": "knowledge_write",
            "description": "Create, revise, append or reversibly archive durable Markdown understanding. YAML carries type, optional title and an authored summary; unknown metadata survives. An active note's summary stays resident in the index; body text never replaces it. Archive hides default navigation rows while direct reads and explicit Presence topics remain available. Restore removes archive metadata, even malformed values in readable YAML, from CURRENT text including subsequent edits. Ordinary writes preserve the lifecycle-owned archive field. Legacy index prose stays until an authored global overview retires it; the improvement backlog keeps its global merge semantics.",
            "parameters": {"type": "object", "properties": {
                "topic": topic, "scope": scope,
                "content": {"type": "string", "description": "Markdown, optionally with YAML frontmatter; supplied fields merge with retained ones. Required except for a summary-only edit; omit for archive/restore. No summary is generated from the body."},
                "mode": {"type": "string", "enum": ["overwrite", "append", "edit", "archive", "restore"], "description": "overwrite replaces the body; append adds text; edit replaces one exact old_str and/or summary. archive/restore preserve body bytes and require a current revision; archive requires reason. overview, patterns and improvement-backlog always stay active. Missing notes are created by overwrite/append only."},
                "reason": {"type": "string", "description": "Required non-empty explanation for archive only; no automatic classification or summarization occurs."},
                "old_str": {"type": "string", "description": "Non-empty exact body substring for mode=edit; it must occur once. content is the replacement, including empty text for a justified deletion. Omit both to change only the summary."},
                "summary": {"type": "string", "description": "mode=edit only: the new authored summary, alone or beside the old_str replacement, in the same revision-checked write."},
                "expected_revision": {"type": "string", "description": "Source revision returned by knowledge_read. Required for overwrite/edit of existing notes and archive/restore. Omit or pass empty only to create a missing note; drift returns the current source without replacing it."},
            }, "required": ["topic"]},
        }, _knowledge_write),
        ToolEntry("knowledge_list", {
            "name": "knowledge_list",
            "description": "List current knowledge with authored summaries and source links. Default active view includes the archive count and address; archived/all reveal archived notes with archive time and reason. Legacy index prose can still mention archived notes until an authored global overview retires it.",
            "parameters": {"type": "object", "properties": {"scope": scope,
                "view": {"type": "string", "enum": ["active", "archived", "all"], "description": "active (default), archived only, or complete inventory."}}, "required": []},
        }, _knowledge_list),
    ] + chronicle_tools()

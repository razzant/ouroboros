"""Knowledge tool adapters over the common linked Markdown source owner."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import List

from ouroboros import knowledge as knowledge_store
from ouroboros.knowledge import INDEX_FILE, OVERVIEW_TOPIC
from ouroboros.knowledge import sanitize_topic as _sanitize_topic
from ouroboros.tools.registry import ToolEntry, ToolContext
from ouroboros.tools.tool_result import ToolResult, _publish_tool_result
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
GLOBAL_ONLY_TOPICS = frozenset({BACKLOG_TOPIC, OVERVIEW_TOPIC, PATTERNS_TOPIC})
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
    ref = note.source_ref()
    text = note.text
    if start_char is None and end_char is None:
        start_char, end_char = 0, len(text)
    if (type(start_char) is not int or type(end_char) is not int
            or not 0 <= start_char <= end_char <= len(text)):
        raise ValueError("Knowledge source range must satisfy 0 <= start_char <= end_char <= complete_chars")
    ref.update(start_char=start_char, end_char=end_char, complete_chars=len(text))
    body = text[start_char:end_char]
    header = "[Knowledge source] " + json.dumps(ref, ensure_ascii=False, sort_keys=True) + "\n"
    if note.parse_error:
        header += "[Metadata unavailable; the complete original Markdown follows.]\n"
    header += "\n"
    return header + body, {
        "knowledge_source": ref, "knowledge_body_start": len(header),
        "knowledge_body_chars": len(body), "knowledge_source_complete": start_char == 0 and end_char == len(text),
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
            status="error", code="TOOL_ARG_ERROR", text=f"⚠️ TOOL_ARG_ERROR: {exc}"))
    except FileNotFoundError:
        elsewhere = ("" if address.scope == "global"
                     else f" A global note may exist: knowledge_read(topic={sanitized!r}, scope='global').")
        return _publish_tool_result(ctx, ToolResult(
            status="ok", code="LEGACY_WARNING", text=f"Topic {topic!r} not found in {address.scope}. Use knowledge_list to see available topics." + elsewhere,
            meta={"knowledge_address": address.as_dict(), "knowledge_missing": True}))
    except (OSError, UnicodeDecodeError) as exc:
        return _publish_tool_result(ctx, ToolResult(
            status="error", code="TOOL_REPORTED_FAILURE", text=f"⚠️ TOOL_ERROR: Knowledge source unavailable: {type(exc).__name__}"))
    return _publish_tool_result(ctx, ToolResult(status="ok", code="OK", text=text, meta=meta))


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



def _knowledge_write(
    ctx: ToolContext, topic: str, content: str, mode: str = "overwrite",
    scope: str = "", expected_revision: str | None = None,
) -> str:
    try:
        sanitized = _sanitize_topic(topic)
        if mode not in ("overwrite", "append") or not isinstance(content, str):
            raise ValueError("content must be Markdown; mode must be overwrite or append")
        if sanitized == BACKLOG_TOPIC:
            from ouroboros.improvement_backlog import backlog_path, merge_backlog_text
            root = _backlog_root(ctx)
            merged = merge_backlog_text(root, content)
            if merged < 0:
                raise ValueError("The improvement-backlog requires parseable ### ibl-<id> blocks with - summary: lines; the global backlog was preserved")
            _record_backlog_history(backlog_path(root), sanitized, mode, str(getattr(ctx, "task_id", "") or ""))
            return f"✅ Knowledge '{sanitized}' merged into the global backlog ({merged} item(s))."
        result = knowledge_store.write_knowledge_note(
            _address(ctx, sanitized, scope), content, mode, expected_revision,
            str(getattr(ctx, "task_id", "") or ""),
        )
    except ValueError as exc:
        return _publish_tool_result(ctx, ToolResult(
            status="error", code="TOOL_ARG_ERROR", text=f"⚠️ TOOL_ARG_ERROR: {exc}"))
    except (OSError, UnicodeDecodeError) as exc:
        return _publish_tool_result(ctx, ToolResult(
            status="error", code="TOOL_REPORTED_FAILURE", text=f"⚠️ TOOL_ERROR: Knowledge write failed: {type(exc).__name__}"))
    meta = {"knowledge_write_reason": result.reason}
    if result.current is not None:
        meta["knowledge_source"] = result.current.source_ref()
    if result.ok:
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
    return _publish_tool_result(ctx, ToolResult(
        status="error", code="TOOL_REPORTED_FAILURE", text=text, meta=meta))


def _knowledge_list(ctx: ToolContext, scope: str = "") -> str:
    try:
        address = _address(ctx, "topic", scope)
        index = address.shelf / INDEX_FILE
        if index.exists():
            return index.read_text(encoding="utf-8")
        rows = knowledge_store.inventory_knowledge(address)
        return (knowledge_store.render_knowledge_index(rows) if rows
                else "Knowledge base is empty. Use knowledge_write to add topics.")
    except ValueError as exc:
        return _publish_tool_result(ctx, ToolResult(
            status="error", code="TOOL_ARG_ERROR", text=f"⚠️ TOOL_ARG_ERROR: {exc}"))


def get_tools() -> List[ToolEntry]:
    topic = {"type": "string", "description": "Shelf-relative topic path without .md; nested paths and Unicode names are supported; no scope prefixes (global/, project/)."}
    scope = {"type": "string", "description": "global or project:<exact project id>. Omitted uses this task's project shelf, otherwise global. Global knowledge remains explicitly reachable from a project. Understanding of people and relationships, and anything that should outlive the project, belongs in global. Reserved topics (improvement-backlog, overview, patterns) always resolve to global."}
    return [
        ToolEntry("knowledge_read", {
            "name": "knowledge_read",
            "description": "Read a complete knowledge note with its canonical address and exact source revision. Follow Markdown links relative to the note's source. Read current understanding before revising it.",
            "parameters": {"type": "object", "properties": {"topic": topic, "scope": scope,
                "start_char": {"type": "integer", "description": "Optional half-open character range start in the exact complete note. Supply both bounds to read a large source in parts."},
                "end_char": {"type": "integer", "description": "Exclusive range end; complete_chars and revision are returned with every view. Omit both bounds for the full note."}}, "required": ["topic"]},
        }, _knowledge_read),
        ToolEntry("knowledge_write", {
            "name": "knowledge_write",
            "description": "Create, revise or append durable understanding in the shared Markdown knowledge corpus. New notes, and legacy notes you meaningfully revise, carry YAML type, optional title and an authored multiline summary, with ordinary Markdown links and source-grounded body; unknown metadata survives. The summary is what stays resident in the index: include it in frontmatter to revise it, while a body-only overwrite keeps the previous frontmatter, including its summary (supplied fields merge with retained ones). Existing legacy notes stay readable. The improvement backlog retains its global merge semantics.",
            "parameters": {"type": "object", "properties": {
                "topic": topic, "scope": scope,
                "content": {"type": "string", "description": "Markdown, optionally with YAML frontmatter. Write understanding and its sources/uncertainty in your own words; no summary is generated from the body."},
                "mode": {"type": "string", "enum": ["overwrite", "append"], "description": "overwrite (default) replaces the body; append adds to the current source. Missing notes are created."},
                "expected_revision": {"type": "string", "description": "Source revision returned by knowledge_read. Required when overwriting an existing note; drift returns the newer source without replacing it."},
            }, "required": ["topic", "content"]},
        }, _knowledge_write),
        ToolEntry("knowledge_list", {
            "name": "knowledge_list",
            "description": "List the selected knowledge shelf with authored summaries and source links. This generated inventory is separate from the shared authored overview.",
            "parameters": {"type": "object", "properties": {"scope": scope}, "required": []},
        }, _knowledge_list),
    ]

"""Persistent topic-based knowledge files with an auto-maintained index."""

import json
import hashlib
import logging
import os
import re
from contextlib import contextmanager
from pathlib import Path
from typing import List

from ouroboros.tools.registry import ToolEntry, ToolContext
from ouroboros.utils import utc_now_iso
from ouroboros.platform_layer import file_lock_exclusive, file_unlock

log = logging.getLogger(__name__)

KNOWLEDGE_DIR = "memory/knowledge"
INDEX_FILE = "index-full.md"
# The immune improvement backlog is ONE global store, never per-project (C10.1).
BACKLOG_TOPIC = "improvement-backlog"


def _backlog_root(ctx: ToolContext) -> Path:
    """Canonical drive root for the global immune backlog. Prefer the canonical
    status root (``budget_drive_root``, set for forked/child drives) so the backlog
    is ONE store that survives forks — never a project-scoped or child-drive copy."""
    return Path(str(getattr(ctx, "budget_drive_root", "") or ctx.drive_root))


def _knowledge_dir(ctx: ToolContext) -> Path:
    """Resolve the knowledge base dir: a per-project store under the CANONICAL data
    dir when the task is project-scoped (Phase 3b), else the canonical
    ``memory/knowledge`` under the task's drive. Project facts thus persist across
    forked/empty child drives and stay isolated from the global memory tree."""
    pid = str(getattr(ctx, "project_id", "") or "").strip()
    if pid:
        from ouroboros.project_facts import project_knowledge_dir

        return project_knowledge_dir(pid)
    return ctx.drive_path(KNOWLEDGE_DIR)

_VALID_TOPIC = re.compile(r'^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,98}[a-zA-Z0-9]$|^[a-zA-Z0-9]$')
_RESERVED = frozenset({"_index", "index-full", "con", "prn", "aux", "nul"})


def _sanitize_topic(topic: str) -> str:
    """Validate a topic name and raise ValueError on bad input."""
    if not topic or not isinstance(topic, str):
        raise ValueError("Topic must be a non-empty string")

    topic = topic.strip()

    if '/' in topic or '\\' in topic or '..' in topic:
        raise ValueError(f"Invalid characters in topic: {topic}")

    if not _VALID_TOPIC.match(topic):
        raise ValueError(f"Invalid topic name: {topic}. Use alphanumeric, underscore, hyphen, dot.")

    if topic.lower() in _RESERVED:
        raise ValueError(f"Reserved topic name: {topic}")

    return topic


def _safe_path(ctx: ToolContext, topic: str) -> tuple[Path, str]:
    """Build a knowledge path and verify containment."""
    sanitized_topic = _sanitize_topic(topic)
    kdir = _knowledge_dir(ctx)
    path = kdir / f"{sanitized_topic}.md"

    resolved = path.resolve()
    kdir_resolved = kdir.resolve()

    try:
        resolved.relative_to(kdir_resolved)
    except ValueError:
        raise ValueError(f"Path escape detected: {topic}")

    return path, sanitized_topic


def _ensure_dir(ctx: ToolContext):
    """Create the knowledge directory."""
    _knowledge_dir(ctx).mkdir(parents=True, exist_ok=True)


@contextmanager
def _knowledge_write_lock(knowledge_dir: Path):
    """Serialize topic and index mutation on one stable directory sidecar."""
    knowledge_dir.mkdir(parents=True, exist_ok=True)
    fd = os.open(knowledge_dir / f"{INDEX_FILE}.lock", os.O_RDWR | os.O_CREAT, 0o644)
    try:
        file_lock_exclusive(fd)
    except Exception:
        os.close(fd)
        raise
    try:
        yield
    finally:
        try:
            file_unlock(fd)
        finally:
            os.close(fd)


def _extract_summary(text: str, max_chars: int = 150) -> str:
    """Extract up to three non-heading snippets for the index."""
    lines = text.strip().split("\n")
    snippets = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        clean = stripped.lstrip("-*").strip().lstrip("#").strip()
        if clean:
            snippets.append(clean)
        if len(snippets) >= 3:
            break

    summary = " | ".join(snippets)
    if len(summary) > max_chars:
        summary = summary[:max_chars - 1] + "…"
    return summary


def _update_index_entry(ctx: ToolContext, topic: str):
    """Update the index entry for one topic."""
    kdir = _knowledge_dir(ctx)
    index_path = kdir / INDEX_FILE
    topic_path = kdir / f"{topic}.md"

    _ensure_dir(ctx)

    if index_path.exists():
        index_content = index_path.read_text(encoding="utf-8")
    else:
        # Seed a FULL index, not just this topic: with the read-path rebuild
        # deleted (#447 J5, "index maintenance stays on the write path"), a
        # one-topic seed would persist a partial index that hides every
        # pre-existing topic file from all later listings.
        seeded = []
        for f in sorted(kdir.glob("*.md")) if kdir.exists() else []:
            if f.name == INDEX_FILE or f.stem == topic:
                continue
            try:
                seeded_topic = _sanitize_topic(f.stem)
            except ValueError:
                continue
            try:
                summary = _extract_summary(f.read_text(encoding="utf-8").strip())
            except Exception:
                # A transient read failure must not hide the topic from every
                # later listing (the persisted index wins over the files): the
                # NAME is known from the filename — keep the entry, mark it.
                log.debug(f"Failed to seed index summary for: {f.stem}", exc_info=True)
                summary = "(summary unavailable at index seed)"
            seeded.append(f"- **{seeded_topic}**: {summary}")
        index_content = "# Knowledge Base Index\n\n" + ("\n".join(seeded) + "\n" if seeded else "")

    lines = index_content.split("\n")
    header_end = 0
    for i, line in enumerate(lines):
        if line.startswith("# "):
            header_end = i + 1
            if i + 1 < len(lines) and lines[i + 1].strip() == "":
                header_end = i + 2
            break

    header = "\n".join(lines[:header_end])
    entries = [line for line in lines[header_end:] if line.strip() and line.strip() != "(empty)"]

    pattern = f"- **{topic}**:"
    entries = [e for e in entries if not e.strip().startswith(pattern)]

    if topic_path.exists():
        try:
            text = topic_path.read_text(encoding="utf-8").strip()
            summary = _extract_summary(text)
            new_entry = f"- **{topic}**: {summary}"
        except Exception:
            log.debug(f"Failed to read knowledge file for index update: {topic}", exc_info=True)
            new_entry = f"- **{topic}**: (unreadable)"

        entries.append(new_entry)
        entries.sort(key=lambda e: e.lower())

    if entries:
        new_index = header.rstrip("\n") + "\n\n" + "\n".join(entries) + "\n"
    else:
        new_index = header.rstrip("\n") + "\n\n(empty)\n"

    temp_path = index_path.with_suffix(".tmp")
    temp_path.write_text(new_index, encoding="utf-8")
    temp_path.replace(index_path)


def _knowledge_read(ctx: ToolContext, topic: str) -> str:
    """Read a knowledge topic."""
    try:
        sanitized_topic = _sanitize_topic(topic)
    except ValueError as e:
        return f"⚠️ Invalid topic: {e}"

    # The improvement backlog always resolves to the ONE global store, regardless
    # of project scope or a forked child drive (C10.1) — never a project copy.
    if sanitized_topic == BACKLOG_TOPIC:
        from ouroboros.improvement_backlog import backlog_path

        path = backlog_path(_backlog_root(ctx))
        if not path.exists():
            return f"Topic '{sanitized_topic}' not found. Use knowledge_list to see available topics."
        return path.read_text(encoding="utf-8")

    try:
        path, sanitized_topic = _safe_path(ctx, topic)
    except ValueError as e:
        return f"⚠️ Invalid topic: {e}"

    if not path.exists():
        return f"Topic '{sanitized_topic}' not found. Use knowledge_list to see available topics."
    return path.read_text(encoding="utf-8")


def _record_backlog_history(backlog_file: Path, topic: str, mode: str, task_id: str) -> None:
    """Audit a backlog write to the GLOBAL knowledge history (C10.1), mirroring the
    generic knowledge-history schema so the backlog's audit trail lives with the
    other global knowledge, not in a project store. Best-effort; never raises."""
    try:
        history_path = backlog_file.parent.parent / "knowledge_history.jsonl"
        new_content = backlog_file.read_text(encoding="utf-8") if backlog_file.exists() else ""
        with open(history_path, "a", encoding="utf-8") as hf:
            hf.write(json.dumps({
                "ts": utc_now_iso(),
                "task_id": task_id,
                "topic": topic,
                "mode": f"{mode}->merge",
                "new_sha256": hashlib.sha256(new_content.encode("utf-8")).hexdigest() if new_content else "",
            }, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _knowledge_write(ctx: ToolContext, topic: str, content: str, mode: str = "overwrite") -> str:
    """Write or append a knowledge topic."""
    try:
        sanitized_topic = _sanitize_topic(topic)
    except ValueError as e:
        return f"⚠️ Invalid topic: {e}"

    if mode not in ("overwrite", "append"):
        return f"⚠️ Invalid mode '{mode}'. Use 'overwrite' or 'append'."

    # The improvement backlog is ONE global, immune store (C10.1 Fix A): route the
    # WHOLE write — not just the path — to the global backlog regardless of project
    # scope or a forked drive, and MERGE non-destructively. Both modes union the
    # written items in (append never truncated; overwrite no longer can wipe the
    # immune backlog). An unparseable write fails CLOSED — the backlog is preserved.
    if sanitized_topic == BACKLOG_TOPIC:
        from ouroboros.improvement_backlog import backlog_path, merge_backlog_text

        root = _backlog_root(ctx)
        merged = merge_backlog_text(root, content)
        if merged < 0:
            return (
                "⚠️ Refused: the improvement-backlog write contained no parseable item "
                "blocks, so the global immune backlog was left intact (never wiped). "
                "Write `### ibl-<id>` blocks with `- summary: …` lines."
            )
        _record_backlog_history(backlog_path(root), sanitized_topic, mode, str(getattr(ctx, "task_id", "") or ""))
        return f"✅ Knowledge '{sanitized_topic}' merged into the global backlog ({merged} item(s))."

    try:
        path, sanitized_topic = _safe_path(ctx, topic)
    except ValueError as e:
        return f"⚠️ Invalid topic: {e}"

    _ensure_dir(ctx)
    with _knowledge_write_lock(_knowledge_dir(ctx)):
        old_content = path.read_text(encoding="utf-8") if path.exists() else ""

        if mode == "append":
            needs_newline = False
            if path.exists() and path.stat().st_size > 0:
                with open(path, "rb") as rf:
                    rf.seek(-1, 2)
                    if rf.read(1) != b"\n":
                        needs_newline = True

            with open(path, "a", encoding="utf-8") as f:
                if needs_newline:
                    f.write("\n")
                f.write(content)
        else:
            path.write_text(content, encoding="utf-8")

        _update_index_entry(ctx, sanitized_topic)
        new_content = path.read_text(encoding="utf-8") if path.exists() else ""

    try:
        history_path = _knowledge_dir(ctx).parent / "knowledge_history.jsonl"
        with open(history_path, "a", encoding="utf-8") as hf:
            hf.write(json.dumps({
                "ts": utc_now_iso(),
                "task_id": str(getattr(ctx, "task_id", "") or ""),
                "topic": sanitized_topic,
                "mode": mode,
                "old_sha256": hashlib.sha256(old_content.encode("utf-8")).hexdigest() if old_content else "",
                "new_sha256": hashlib.sha256(new_content.encode("utf-8")).hexdigest() if new_content else "",
                "old_content": old_content,
                "new_content": new_content,
            }, ensure_ascii=False) + "\n")
    except Exception:
        pass

    try:
        journal_path = _knowledge_dir(ctx).parent / "knowledge_journal.jsonl"
        total_kb = 0
        knowledge_dir = _knowledge_dir(ctx)
        if knowledge_dir.exists():
            for f in knowledge_dir.iterdir():
                if f.is_file() and f.suffix == ".md":
                    total_kb += f.stat().st_size / 1024
        entry = {
            "ts": utc_now_iso(),
            "topic": sanitized_topic,
            "mode": mode,
            "file_kb": path.stat().st_size / 1024,
            "total_knowledge_kb": round(total_kb, 2),
        }
        with open(journal_path, "a", encoding="utf-8") as jf:
            jf.write(json.dumps(entry) + "\n")
    except Exception:
        pass

    return f"✅ Knowledge '{sanitized_topic}' saved ({mode})."


def _knowledge_list(ctx: ToolContext) -> str:
    """List knowledge topics with summaries."""
    kdir = _knowledge_dir(ctx)
    index_path = kdir / INDEX_FILE

    if index_path.exists():
        return index_path.read_text(encoding="utf-8")

    # No index: render the listing IN MEMORY from the topic files. knowledge_list
    # is registered read-only (safety.py POLICY_SKIP) and granted to children that
    # may not write cognitive memory — the old on-miss index rebuild here created
    # the knowledge dir, a lock sidecar and index-full.md on a pure read. Index
    # maintenance stays on the write path (_update_index_entry).
    entries = []
    for f in sorted(kdir.glob("*.md")) if kdir.exists() else []:
        if f.name == INDEX_FILE:
            continue
        try:
            topic = _sanitize_topic(f.stem)
        except ValueError:
            continue
        try:
            summary = _extract_summary(f.read_text(encoding="utf-8").strip())
            entries.append(f"- **{topic}**: {summary}")
        except Exception:
            log.debug(f"Failed to read knowledge file for listing: {topic}", exc_info=True)
            entries.append(f"- **{topic}**: (unreadable)")

    if entries:
        return "# Knowledge Base Index\n\n" + "\n".join(entries) + "\n"
    return "Knowledge base is empty. Use knowledge_write to add topics."


def get_tools() -> List[ToolEntry]:
    return [
        ToolEntry("knowledge_read", {
            "name": "knowledge_read",
            "description": "Read a topic from the persistent knowledge base on Drive. On a project-scoped task, reads from that project's per-project facts store (isolated from global knowledge).",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "Topic name (alphanumeric, hyphens, underscores). E.g. 'browser-automation', 'git-recipes'"
                    }
                },
                "required": ["topic"]
            },
        }, _knowledge_read),
        ToolEntry("knowledge_write", {
            "name": "knowledge_write",
            "description": "Write or append to a knowledge topic. Use for recipes, gotchas, patterns learned from experience. On a project-scoped task, reads/writes are automatically scoped to that project's per-project facts store (isolated from global knowledge).",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "Topic name (alphanumeric, hyphens, underscores)"
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write (markdown)"
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["overwrite", "append"],
                        "description": "Write mode: 'overwrite' (default) or 'append'"
                    }
                },
                "required": ["topic", "content"]
            },
        }, _knowledge_write),
        ToolEntry("knowledge_list", {
            "name": "knowledge_list",
            "description": "List all topics in the knowledge base with summaries. On a project-scoped task, lists only the current project's facts store.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            },
        }, _knowledge_list),
    ]

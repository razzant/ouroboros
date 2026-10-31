"""Thin per-project facts store (Phase 3b).

A project-scoped task (an external/workspace task, or one given an explicit
``project_id``) keeps its learned FACTS in a per-project knowledge store that is:

- under the CANONICAL data dir (``config.DATA_DIR/projects/<id>/knowledge``), NOT
  a task's child drive — so it persists across forked/empty runs;
- OUTSIDE ``memory/knowledge/**`` and any ``_copy_stable_memory`` path — so it
  never leaks into the forked seed or another project (red-team R3.1/guard #2);
- facts only — identity and the scratchpad stay canonical and are written from
  any room through their own tools; there is no per-project copy of either.

This is a thin SSOT helper, NOT a parallel memory subsystem (P7): the existing
knowledge tool + context loader simply redirect their base dir when a task is
project-scoped, and the post-task canonical dual-run is suppressed for such tasks
so the automatic post-task writers cannot contaminate the global memory.
"""
from __future__ import annotations

import hashlib
import pathlib
import re
from typing import Any, Dict

_SAFE = re.compile(r"[^a-zA-Z0-9_.-]")
# A character a slug cannot carry: whitespace maps to "-" deterministically, so it
# is not loss; anything else disappears and two different names can collapse onto
# one id (see project_id_from_display_name).
_LOSSY = re.compile(r"[^a-zA-Z0-9_.\-\s]")
# Windows reserved device names (case-insensitive, incl. extension variants like
# "con.md"): never allow these as a project dir component.
_RESERVED_NAMES = frozenset(
    {"con", "prn", "aux", "nul"}
    | {f"com{i}" for i in range(1, 10)}
    | {f"lpt{i}" for i in range(1, 10)}
)


def sanitize_project_id(value: Any) -> str:
    """Return a filesystem-safe project id (alphanumeric/_/-/., <=64 chars), or ""
    if unusable (empty, or a Windows reserved device name)."""
    pid = _SAFE.sub("-", str(value or "").strip())[:64].strip("-.")
    if pid.lower().split(".", 1)[0] in _RESERVED_NAMES:
        return ""
    # Canonical lowercase: on case-insensitive filesystems (macOS/Windows) "Proj"
    # and "proj" would otherwise alias the same store and break isolation.
    return pid.casefold()


_MINTED_PROJECT_ID = re.compile(r"proj_[0-9a-f]+")


def project_display_name(name: Any, project_id: Any) -> str:
    """The Project's own name, or "" for a row that was never named.

    A never-named row carries its minted ``proj_<hex>`` id as the name (the two
    minters below); only that shape is a non-name. An owner's own ``blog`` has
    the id ``blog`` too, and it IS the name (DESIGN "References and actions";
    the browser twin is ``project_reference.js`` ``MINTED_ID``).
    """
    text = " ".join(str(name or "").split())
    return "" if not text or (text == str(project_id or "") and _MINTED_PROJECT_ID.fullmatch(text)) else text


def project_id_from_display_name(value: Any) -> str:
    """Derive a filesystem-clean project id from a human DISPLAY name, ALWAYS
    yielding a usable id for any non-blank name. The sanitized slug is used when it
    has clean characters; otherwise a deterministic hash id (so a Cyrillic-only or
    emoji-only name like 'динозавры' still creates a project instead of failing —
    the Russian-speaking owner's common case). The real display name is stored
    separately on the registry, so the user always sees their own name, never the
    id. Returns "" only for a truly empty name.

    Two rules keep a LOSSY name from producing a misleading or COLLIDING id: runs
    of ``-`` collapse (a Cyrillic word between two Latin ones minted
    ``mlconf--------------------ouroboros``), and when the name held any character
    the slug could not carry, a short digest of the whole raw name is appended — two
    different Cyrillic titles otherwise normalized onto the same id and silently
    shared one project. Only NEWLY minted ids are affected: ``sanitize_project_id``
    is untouched, so every existing id, lookup and derived project chat stays
    exactly as it is."""
    raw = str(value or "").strip()
    if not raw:
        return ""
    slug = re.sub(r"-{2,}", "-", sanitize_project_id(raw)).strip("-.")
    if not slug:
        return "proj_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
    if _LOSSY.search(raw):
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:8]
        # The slug never starts with "-" or ".", so the truncated head stays usable.
        slug = f"{slug[:63 - len(digest)].strip('-.')}-{digest}"
    return slug


# A NEW project folder keeps its human name (#1308): only what a supported
# filesystem cannot hold is replaced — path separators, the Windows-forbidden set,
# control characters — and whitespace runs become "-". Leading/trailing dots,
# spaces and dashes go (hidden, "."/"..", Windows trailing rules). Bounded by
# characters AND UTF-8 bytes; a cut name carries a digest of the RAW name so two
# long names with one prefix stay distinguishable (#850 rule for ids).
_FOLDER_FORBIDDEN = re.compile(r'[<>:"/\\|?*]')
_FOLDER_MAX_CHARS = 64
_FOLDER_MAX_BYTES = 180  # leaves room for a "_N" collision suffix under 255 bytes


def project_folder_basename(display_name: Any) -> str:
    """Readable basename for a NEWLY created project folder, or "" when nothing
    usable remains. NFC normalization happens here, at creation only: existing
    folders, ids and refs are never re-derived or compared through it."""
    import unicodedata

    raw = unicodedata.normalize("NFC", str(display_name or "").strip())
    name = "".join("_" if unicodedata.category(ch) in {"Cc", "Cs"} else ch for ch in raw)
    name = re.sub(r"\s+", "-", _FOLDER_FORBIDDEN.sub("_", name)).strip(" .-")
    if len(name) > _FOLDER_MAX_CHARS or len(name.encode("utf-8")) > _FOLDER_MAX_BYTES:
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:8]
        head = name[:_FOLDER_MAX_CHARS - len(digest) - 1]
        while len(head.encode("utf-8")) > _FOLDER_MAX_BYTES - len(digest) - 1:
            head = head[:-1]
        head = head.rstrip(" .-")
        name = f"{head}-{digest}" if head else digest
    if name.split(".", 1)[0].casefold() in (_RESERVED_NAMES | {f"{prefix}{digit}" for prefix in ("com", "lpt") for digit in "¹²³"}):
        stem, dot, rest = name.partition(".")
        name = f"{stem}_{dot}{rest}"
    return name


def explicit_project_id_ok(raw: Any) -> bool:
    """True if an EXPLICIT project id is already filesystem-clean (no silent
    normalization). The gateway rejects explicit ids that fail this, so two
    different inputs can never collapse to the same store and an unusable id never
    silently falls back to canonical memory."""
    s = str(raw or "")
    return bool(s) and s == s.strip() and s == sanitize_project_id(s)


def _normalized_workspace(workspace: str) -> str:
    """normcase(realpath) of a workspace path — the canonical key used for BOTH the
    derived-id hash and the registry-first match, so a folder maps to ONE identity on
    case-insensitive filesystems (identity on case-sensitive Linux)."""
    import os

    return os.path.normcase(str(pathlib.Path(workspace).resolve(strict=False)))


def _registered_project_for_workspace(canon_workspace: str) -> str:
    """v6.58.0 (slice 0) — registry-first: return the id of a REGISTERED project whose
    normalized working_dir equals ``canon_workspace``, else "". So a `/api/tasks`
    workspace task and its owner project resolve to ONE id — one lease lane, one
    knowledge store — instead of the folder acquiring a second `proj_<hash>` identity.
    Anchored at the canonical DATA_DIR (where projects.json lives). Fail-open."""
    if not canon_workspace:
        return ""
    try:
        from ouroboros.config import DATA_DIR
        from ouroboros.projects_registry import list_projects

        for project in list_projects(DATA_DIR):
            wd = str(project.get("working_dir") or "").strip()
            if wd and _normalized_workspace(wd) == canon_workspace:
                return str(project.get("id") or "")
    except Exception:
        return ""
    return ""


def _bound_project_id(task_id: str) -> str:
    """Project a task is DURABLY bound to, read at the canonical DATA_DIR, else "".

    Fail-OPEN, exactly like ``_registered_project_for_workspace``: this runs on
    the hot resolution path of every context build, and an unreadable bindings
    store must not stop the work. The one caller that AUTHORIZES creating a
    project (``ensure_project_scope``) reads the same binding strictly instead.
    """
    if not task_id:
        return ""
    try:
        from ouroboros.config import DATA_DIR
        from ouroboros.projects_registry import project_id_for_task

        return sanitize_project_id(project_id_for_task(DATA_DIR, task_id))
    except Exception:
        return ""


def placed_project_id(ctx: Any) -> Any:
    """Resolve the current memory room without changing execution context fields."""
    from ouroboros.projects_registry import project_id_for_task
    from ouroboros.tool_access import canonical_data_root

    tid = str(getattr(ctx, "task_id", "") or "")
    bound = project_id_for_task(canonical_data_root(ctx), tid, strict=True) if tid else ""
    return bound or getattr(ctx, "project_id", "")


def resolve_project_id(task: Dict[str, Any]) -> str:
    """Resolve a task's project id (S7): the DURABLE binding wins (owner decision
    B4=A - a "turn into project" conversion never reaches a running worker, so the
    task dict can still say nothing while the task belongs to a project); else an
    explicit ``project_id``; else, for a workspace task, a REGISTERED project bound
    to that folder (v6.58.0 registry-first) or a stable hash of the workspace path;
    else ``""`` (not project-scoped — canonical memory, unchanged behavior)."""
    if not isinstance(task, dict):
        return ""
    bound = _bound_project_id(str(task.get("id") or task.get("task_id") or "").strip())
    if bound:
        return bound
    pid = sanitize_project_id(task.get("project_id"))
    if pid:
        return pid
    # Subagents inherit the parent's scope EXPLICITLY (carried on the child task);
    # never re-derive from the child's (possibly acting) workspace, which would
    # mismatch the forked seed prepared at schedule time for an unscoped parent.
    if str(task.get("delegation_role") or "") == "subagent":
        return ""
    from ouroboros.dialogue_provenance import presence_metadata_binding

    if presence_metadata_binding(task.get("metadata")) is not None:
        # Presence (a speaker, or work acting for its binding) changes file/process cwd,
        # not its canonical memory scope.
        return ""
    workspace = str(task.get("workspace_root") or "").strip()
    if workspace:
        canon = _normalized_workspace(workspace)
        # Registry-first: one folder = one project identity (and thus one lease lane).
        registered = _registered_project_for_workspace(canon)
        if registered:
            return registered
        digest = hashlib.sha256(canon.encode("utf-8")).hexdigest()[:12]
        return f"proj_{digest}"
    return ""


def project_store_access_block(rel_path: Any) -> "str | None":
    """Deny message if ``rel_path`` targets the per-project store (``projects/<id>/``),
    else None. Generic data tools use this so the store is reachable ONLY via the
    project-scoped tools (no cross-project peeking). The path is normalized
    first (collapsing ``.``/``..`` and backslashes) so traversal/`./`-prefixed forms
    cannot bypass the check."""
    import os as _os

    raw = str(rel_path or "").replace("\\", "/").strip()
    raw = re.sub(r"^[a-zA-Z]:", "", raw)  # strip a Windows drive letter before checking
    normalized = _os.path.normpath(raw).replace("\\", "/").lstrip("/")
    first = normalized.split("/", 1)[0].casefold() if normalized else ""
    if first == "projects":
        return ("⚠️ ACCESS_DENIED: the per-project facts store (projects/<id>/) is not "
                "reachable via generic data tools. " + _project_store_reader_hint(normalized.split("/")[1:]))
    return None


def _project_store_reader_hint(parts: list) -> str:
    """The scoped tools whose address space holds ``projects/<parts>``, read from the
    layout the path helpers below own (#1612). One text answers reads, writes, listings
    and edits, so it names the exact read/write pair and never more access than those
    tools already grant; anything they do not address is said plainly."""
    from ouroboros.knowledge import GLOBAL_ADDRESS_TOPICS, INDEX_FILE, sanitize_topic

    pid, rest = (parts[0], "/".join(parts[1:])) if parts else ("", "")
    if rest and explicit_project_id_ok(pid):
        own, scope = _project_store_root(pid), f"scope='project:{pid}'"
        shelf = project_knowledge_dir(pid).relative_to(own).as_posix()
        if rest in (shelf, f"{shelf}/{INDEX_FILE}"):
            return f"This is a project knowledge shelf: knowledge_list({scope}) lists it."
        if rest.startswith(shelf + "/") and rest.endswith(".md") and len(rest) > len(shelf) + 4:
            try:  # the knowledge owner's own topic rules decide what a call would address
                topic = sanitize_topic(rest[len(shelf) + 1:-3])
            except ValueError:
                return f"This file is not addressable as a knowledge topic; knowledge_list({scope}) lists the shelf."
            if topic in GLOBAL_ADDRESS_TOPICS:
                return (f"Topic {topic!r} is global-only: knowledge_read/knowledge_write(topic={topic!r}) read and "
                        "replace the global note even with a project scope, never this project copy; "
                        f"knowledge_list({scope}) may still list it.")
            return (f"This is a project knowledge topic: knowledge_read(topic={topic!r}, {scope}) reads it and "
                    f"knowledge_write(topic={topic!r}, {scope}, content=..., expected_revision=...) writes it; "
                    "expected_revision is the revision knowledge_read returned, omitted only to create a missing note.")
        part = {
            project_workpad_path(pid).relative_to(own).as_posix(): (
                "workpad", f"workpad_read(project_id='{pid}')",
                f"workpad_write(project_id='{pid}', content=...) replaces it"),
            project_journal_path(pid).relative_to(own).as_posix(): (
                "journal", f"journal_read(project_id='{pid}')",
                f"journal_write(project_id='{pid}', kind=..., text=...) appends to it"),
            project_reflections_path(pid).relative_to(own).as_posix(): (
                "reflection log", f"journal_read(project_id='{pid}', source='reflections')", ""),
        }.get(rest)
        if part:
            noun, reader, writer = part
            return (f"This is project {pid}'s {noun}: {reader} reads it" + (f" and {writer}" if writer else "")
                    + ". Outside its own project a task reads it only as a root task"
                    + ("; it writes it only as a root task placed in no project (such as Main)."
                       if writer else "; no tool writes it."))
    tools = ("knowledge_list/knowledge_read/knowledge_write (knowledge), workpad_read/workpad_write "
             "(workpad) and journal_read/journal_write (journal; source='reflections' reads reflections)")
    if not rest:
        return f"Its scoped tools are {tools}."
    return f"No scoped reader or writer addresses this path; the store's scoped tools are {tools}."


def filter_out_project_store(base_rel: Any, names: Any) -> list:
    """Drop entries that would expose the per-project store from a generic listing.
    The combined path is handed to project_store_access_block, which normalizes
    `.`/`..` itself, so a traversal base like ``logs/..`` cannot smuggle ``projects``."""
    base = str(base_rel or "").rstrip("/")
    out = []
    for n in names:
        rel = (base + "/" + str(n)) if base else str(n)
        if not project_store_access_block(rel):
            out.append(n)
    return out


def project_knowledge_dir(project_id: str) -> pathlib.Path:
    """Absolute path to a project's knowledge dir under the canonical data dir."""
    from ouroboros.config import DATA_DIR

    return pathlib.Path(DATA_DIR) / "projects" / sanitize_project_id(project_id) / "knowledge"


def _project_store_root(project_id: str) -> pathlib.Path:
    from ouroboros.config import DATA_DIR

    return pathlib.Path(DATA_DIR) / "projects" / sanitize_project_id(project_id)


def project_journal_path(project_id: str) -> pathlib.Path:
    """Append-only project journal (milestones: start/blocked/checkpoint/done).

    JSONL rows ``{ts, kind, text, task_id}`` — the durable per-project memory
    the agent (and the absorption digest) reads back. Same store subtree as
    project knowledge, so ``project_store_access_block`` already guards it.
    """
    return _project_store_root(project_id) / "journal.jsonl"


def project_workpad_path(project_id: str) -> pathlib.Path:
    """Free-form per-project working notes (markdown scratchpad)."""
    return _project_store_root(project_id) / "workpad.md"


def project_reflections_path(project_id: str) -> pathlib.Path:
    """Durable per-project execution reflections (full text lives HERE for
    project-scoped roots; the canonical log carries a bounded pointer row)."""
    return _project_store_root(project_id) / "logs" / "task_reflections.jsonl"

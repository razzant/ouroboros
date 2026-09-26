"""Thin per-project journal/workpad tools (multi-project, v6.32.0).

The journal is the project's durable milestone memory (start / blocked /
checkpoint / done / note rows); the workpad is a free-form scratch page. Both
live in the per-project store (``data/projects/<id>/``), which generic data
tools cannot reach (``project_store_access_block``) — these scoped tools are
the only write path, exactly like project knowledge.

Tools resolve the project from the CURRENT task (``ctx.project_id``); an
explicit ``project_id`` argument lets the main-chat agent annotate a specific
project (e.g. when curating from the штаб).
"""

from __future__ import annotations

import hashlib
import json
import logging
import pathlib
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Tuple

from ouroboros.project_facts import (
    project_journal_path,
    project_workpad_path,
    sanitize_project_id,
    explicit_project_id_ok,
)
from ouroboros.dialogue_provenance import is_presence_task, presence_caller_binding
from ouroboros.focus import normalize_focus
from ouroboros.tools.registry import ToolContext, ToolEntry
from ouroboros.utils import (
    append_jsonl,
    jsonl_generation_signature,
    utc_now_iso,
)

log = logging.getLogger(__name__)

_JOURNAL_KINDS = ("start", "checkpoint", "blocked", "done", "note")
_MAX_TEXT_CHARS = 4000
_WORKPAD_MAX_BYTES = 256 * 1024


def _scope_authority(ctx: ToolContext) -> tuple[str, Dict[str, Any]]:
    metadata = getattr(ctx, "task_metadata", {})
    metadata = metadata if isinstance(metadata, dict) else {}
    contract = getattr(ctx, "task_contract", {})
    contract = contract if isinstance(contract, dict) else {}
    lineage = contract.get("lineage") if isinstance(contract.get("lineage"), dict) else {}
    task = {"metadata": metadata, "task_contract": contract,
            "_presence_turn": bool(metadata.get("_presence_turn") or getattr(ctx, "_presence_turn", False)),
            "_presence_origin": getattr(ctx, "_presence_origin", None)}
    parent_task_id = str(lineage.get("parent_task_id") or metadata.get("parent_task_id") or "").strip()
    delegation_role = str(lineage.get("delegation_role") or metadata.get("delegation_role") or "").strip()
    root_task_id = str(lineage.get("root_task_id") or metadata.get("root_task_id") or "").strip()
    child = bool(parent_task_id) or delegation_role == "subagent"
    if is_presence_task(task) or presence_caller_binding(ctx) is not None:  # a speaker, or acting for its binding
        return "presence", metadata
    if child:
        return "child", metadata
    # Root markers are host admission facts. A drive path or a missing metadata
    # object is not evidence of root authority: test-shaped fallbacks here used
    # to let an arbitrary scoped actor read another project's journal.
    root = bool(getattr(ctx, "is_direct_chat", False)) or delegation_role == "root" or bool(root_task_id)
    return ("root" if root else "scoped"), metadata


def _resolve_project_id(ctx: ToolContext, explicit: Any, *, write: bool = False) -> tuple[str, str]:
    own_value = getattr(ctx, "project_id", "")
    if own_value not in (None, "") and not isinstance(own_value, str):
        return "", "⚠️ TOOL_ARG_ERROR: current project scope is malformed"
    if explicit not in (None, "") and not isinstance(explicit, str):
        return "", "⚠️ TOOL_ARG_ERROR: project_id is malformed"
    own_raw = str(own_value or "").strip()
    requested_raw = str(explicit or "").strip()
    own = sanitize_project_id(own_raw) if own_raw else ""
    if requested_raw and not explicit_project_id_ok(requested_raw):
        return "", "⚠️ TOOL_ARG_ERROR: project_id is malformed"
    requested = sanitize_project_id(requested_raw) if requested_raw else ""
    authority, _metadata = _scope_authority(ctx)
    if requested_raw and not requested:
        return "", "⚠️ TOOL_ARG_ERROR: project_id is malformed"
    if own_raw and not own:
        return "", "⚠️ TOOL_ARG_ERROR: current project scope is malformed"
    if own and requested and requested != own:
        if write:
            return "", "⚠️ TOOL_FORBIDDEN: foreign project writes are refused; write only the current project"
        if authority != "root":
            return "", "⚠️ TOOL_FORBIDDEN: this actor may read only its current project"
        return requested, ""
    if requested and not own and authority != "root":
        return "", "⚠️ TOOL_FORBIDDEN: restricted actors may not access a foreign project"
    return own or requested, ""


def _authorized_project_id(ctx: ToolContext, explicit: Any) -> str:
    """Compatibility resolver for the current scope (read semantics)."""
    return _resolve_project_id(ctx, explicit, write=False)[0]


def _journal_write(ctx: ToolContext, kind: str, text: str, project_id: str = "") -> str:
    pid, scope_error = _resolve_project_id(ctx, project_id, write=True)
    if scope_error:
        return scope_error + " (journal_write)"
    if not pid:
        return ("⚠️ TOOL_ARG_ERROR (journal_write): no project scope — this task is not "
                "project-scoped and no explicit project_id was given.")
    kind_norm = str(kind or "note").strip().lower()
    if kind_norm not in _JOURNAL_KINDS:
        return f"⚠️ TOOL_ARG_ERROR (journal_write): kind must be one of {_JOURNAL_KINDS}"
    body = str(text or "").strip()
    if not body:
        return "⚠️ TOOL_ARG_ERROR (journal_write): text is required"
    # The journal is durable cognitive memory — never silently slice a stored
    # entry. Reject over-limit writes (same contract as workpad_write) so the
    # agent shortens the milestone or moves detail to the workpad/knowledge.
    if len(body) > _MAX_TEXT_CHARS:
        return (f"⚠️ TOOL_ARG_ERROR (journal_write): entry exceeds {_MAX_TEXT_CHARS} chars "
                f"({len(body)}) — a journal entry is a milestone note; keep it short and "
                "move long detail to workpad_write or knowledge_write.")
    path = project_journal_path(pid)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "ts": utc_now_iso(),
        "kind": kind_norm,
        "text": body,
        "task_id": str(getattr(ctx, "task_id", "") or ""),
    }
    append_jsonl(path, row)
    try:
        from ouroboros.config import DATA_DIR
        from ouroboros.projects_registry import touch_project

        # Registry lives on the CANONICAL data dir (like project_journal_path),
        # not a forked child drive — touching ctx.drive_root would scatter
        # stray projects.json files onto subagent worktrees.
        touch_project(pathlib.Path(DATA_DIR), pid)
    except Exception:
        log.debug("journal touch_project failed", exc_info=True)
    return f"OK: journal[{pid}] += {kind_norm} entry ({len(body)} chars)."


def append_journal_milestone(
    project_id: str, kind: str, text: str, task_id: str = "", extra: dict | None = None,
) -> None:
    """Append an AUTOMATIC project journal milestone (e.g. task-completion 'letters
    home'), enforcing the SAME durable per-row contract as the journal_write tool.

    The tool REJECTS over-limit input (it teaches the agent to keep milestones
    short); an automatic milestone MUST be recorded, so when the composed text
    exceeds ``_MAX_TEXT_CHARS`` it is bounded with a VISIBLE pointer instead of
    being silently sliced or dropped (the full text always survives in the task's
    task_results and the consciousness digest). Centralizing here keeps every
    project-journal append on one bounded path (no raw append_jsonl elsewhere)."""
    pid = sanitize_project_id(project_id)
    if not pid:
        return
    raw_kind = str(kind or "").strip().lower()
    kind_norm = raw_kind or "note"
    if kind_norm not in _JOURNAL_KINDS:
        # Fail LOUD but never LOSE the entry: an explicitly-passed unknown kind is a caller
        # bug worth surfacing, yet the milestone must still be durably recorded (as a note)
        # rather than silently dropped. (An omitted/empty kind defaults to note quietly.)
        log.warning(
            "append_journal_milestone: unknown kind %r recorded as 'note' (project=%s, task=%s)",
            raw_kind, pid, str(task_id or ""),
        )
        kind_norm = "note"
    body = str(text or "").strip()
    if not body:
        return
    if len(body) > _MAX_TEXT_CHARS:
        keep = _MAX_TEXT_CHARS - 80
        body = body[:keep] + f"… [+{len(body) - keep} chars; full text in this task's task_results / digest]"
    path = project_journal_path(pid)
    path.parent.mkdir(parents=True, exist_ok=True)
    append_jsonl(path, {
        "ts": utc_now_iso(),
        "kind": kind_norm,
        "text": body,
        "task_id": str(task_id or ""),
        # Optional typed payload (e.g. the work-location row's path/sha facts);
        # the reserved row keys always win.
        **{k: v for k, v in dict(extra or {}).items()
           if k not in {"ts", "kind", "text", "task_id"}},
    })
    try:
        from ouroboros.config import DATA_DIR
        from ouroboros.projects_registry import touch_project

        touch_project(pathlib.Path(DATA_DIR), pid)
    except Exception:
        log.debug("append_journal_milestone touch_project failed", exc_info=True)


def _record_work_location(project_id: str, task: dict) -> None:
    """Q8: ONE typed "work lives at <path> @ <sha>" row when the finished task's
    effective working tree is NOT the project's registered working_dir (or the
    registry has none). Continuation promotions read the registry/journal —
    without this row an off-registry tree is invisible to every later task (the
    saga rebuilt a finished game because nothing durable said where the first
    build lived). Uses only facts the task record already holds; never spawns git.
    The sha, when present, is the admission-time preflight head — a tree
    identifier, not a claim about the final commit."""
    meta = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
    workspace = str(task.get("workspace_root") or meta.get("workspace_root") or "").strip()
    if not workspace:
        return
    registered = ""
    try:
        from ouroboros.config import DATA_DIR
        from ouroboros.projects_registry import get_project

        registered = str(
            (get_project(pathlib.Path(DATA_DIR), sanitize_project_id(project_id)) or {})
            .get("working_dir") or ""
        ).strip()
    except Exception:
        log.debug("work-location registry lookup failed", exc_info=True)
    try:
        same = bool(registered) and (
            pathlib.Path(registered).resolve(strict=False)
            == pathlib.Path(workspace).resolve(strict=False)
        )
    except (OSError, ValueError):
        same = registered == workspace
    if same:
        return
    preflight = meta.get("workspace_preflight") if isinstance(meta.get("workspace_preflight"), dict) else {}
    git = preflight.get("git") if isinstance(preflight.get("git"), dict) else {}
    sha = str(git.get("head") or "").strip()
    append_journal_milestone(
        project_id,
        "note",
        f"work lives at {workspace}" + (f" @ {sha}" if sha else ""),
        task_id=str(task.get("id") or ""),
        extra={"type": "work_location", "path": workspace, "sha": sha,
               "registered_working_dir": registered},
    )


def record_project_last_result(project_id: str, task_id: str, drive_root: Any) -> None:
    """Stamp the project's durable last-result pointer (read first by
    ``_latest_project_task_result``). THE one writer of that pointer, shared by the
    pooled-task finalization below and the project room's direct-chat root - which
    writes a durable result carrying ``project_id`` but no letters home, so without
    this the per-project fallback for "continue from this result" stayed empty.

    A split-drive task's canonical copy-back may land moments later; the reader
    validates the pointed file and falls back to the scan. Fail-soft."""
    if drive_root is None or not str(task_id or "").strip() or not str(project_id or "").strip():
        return
    try:
        from ouroboros.projects_registry import update_project

        update_project(drive_root, project_id, last_task_result_id=str(task_id))
    except Exception:
        log.debug("project last-task-result pointer update failed", exc_info=True)


def record_task_finalization(
    project_id: str, task: dict, *, objective: str, kind: str, exec_status: str,
    drive_root: Any = None,
) -> None:
    """One seam for a project root's durable "letters home" at finalization:
    the task-finished milestone, the off-registry work-location row (Q8), the
    registry last-result pointer, and — for the swarm ROOT (no parent) — the
    ephemeral tree-ledger coordination mirror (see
    mirror_tree_coordination_to_journal). Fail-soft per row."""
    tid = str(task.get("id") or "")
    try:
        append_journal_milestone(
            project_id, kind, f"Task finished ({exec_status}): {objective}", task_id=tid,
        )
    except Exception:
        log.debug("project journal task-done entry failed", exc_info=True)
    is_root = not str(task.get("parent_task_id") or "").strip() and str(
        task.get("delegation_role") or "") != "subagent"  # the door's own child predicate
    if is_root:
        # The pointer answers "continue from here" for the ROOM, so only a ROOT may
        # stamp it: a child finalizing after its root moved the room's single
        # candidate onto work no owner ever addressed - a helper the hint never
        # offers (the mirror below is root-only for the same reason).
        record_project_last_result(project_id, tid, drive_root)
    try:
        _record_work_location(project_id, task)
    except Exception:
        log.debug("project journal work-location entry failed", exc_info=True)
    if is_root:
        try:
            mirror_tree_coordination_to_journal(
                project_id, str(task.get("root_task_id") or tid), task_id=tid,
            )
        except Exception:
            log.debug("project journal swarm-coordination mirror failed", exc_info=True)


_TREE_MIRROR_KINDS = {
    # task-tree ledger kind -> durable journal kind. Only the high-signal coordination
    # survives: attention beacons + interface contracts. The low-signal coordination
    # (fact/note/decision) and routine progress (milestone/partial_finding) are NOT
    # mirrored — the journal stays a curated durable record, not a tree echo.
    "blocker": "blocked",
    "question": "note",
    "interface_contract": "note",
    "contract": "note",
}


def mirror_tree_coordination_to_journal(project_id: str, root_id: str, task_id: str = "") -> None:
    """F2 (v6.39): mirror the EPHEMERAL task-tree ledger's durable-worthy swarm coordination
    (attention beacons blocker/question/interface_contract + interface contracts) into the
    DURABLE project journal, so a swarm's decisions/blockers survive the tree's GC. Call once
    on the swarm ROOT's terminal (not per sibling) to avoid re-mirroring the same rows.
    Fail-soft and bounded (each row goes through the same per-row journal contract)."""
    pid = sanitize_project_id(project_id)
    rid = str(root_id or "").strip()
    if not pid or not rid:
        return
    try:
        from ouroboros.task_tree_ledger import tree_ledger_rows
        rows = tree_ledger_rows(rid)
    except Exception:
        log.debug("mirror_tree_coordination_to_journal read failed", exc_info=True)
        return
    for r in rows:
        kind = str(r.get("kind") or "").strip().lower()
        journal_kind = _TREE_MIRROR_KINDS.get(kind)
        if not journal_kind:
            continue
        text = str(r.get("text") or "").strip()
        if not text:
            continue
        who = str(r.get("role") or "") or str(r.get("task_id") or "")[:8]
        append_journal_milestone(pid, journal_kind, f"[swarm {kind}] ({who}): {text}", task_id=task_id)


def _journal_snapshot(source: Dict[str, Any], project_id: str) -> str:
    payload = {
        "schema_version": 1,
        "project_id": project_id,
        "source": source,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _journal_snapshot_rows(
    path: pathlib.Path,
    project_id: str,
) -> tuple[List[Dict[str, Any]], str, bool, int]:
    """Capture one stateless journal generation, retrying one concurrent append."""
    rows: List[Dict[str, Any]] = []
    snapshot = ""
    unreadable = 0
    for _attempt in range(2):
        before = jsonl_generation_signature(path)
        rows = []
        unreadable = 0
        try:
            with path.open("rb") as handle:
                for raw in handle:
                    try:
                        line = raw.decode("utf-8").strip()
                    except UnicodeDecodeError:
                        unreadable += 1
                        continue
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except (json.JSONDecodeError, ValueError):
                        unreadable += 1
                        continue
                    if not isinstance(entry, dict):
                        unreadable += 1
                        continue
                    rows.append(entry)
        except OSError:
            unreadable += 1
        after = jsonl_generation_signature(path)
        snapshot = _journal_snapshot(after, project_id)
        if before and before == after:
            return rows, snapshot, True, unreadable
    return rows, snapshot, False, unreadable


def _journal_read(
    ctx: ToolContext,
    project_id: str = "",
    limit: int = 30,
    offset: int = 0,
    snapshot: str = "",
) -> str:
    pid, scope_error = _resolve_project_id(ctx, project_id, write=False)
    if scope_error:
        return scope_error + " (journal_read)"
    if not pid:
        return ("⚠️ TOOL_ARG_ERROR (journal_read): no project scope — this task is not "
                "project-scoped and no explicit project_id was given.")
    path = project_journal_path(pid)
    if not path.is_file():
        if str(snapshot or "").strip():
            return (
                "JOURNAL_READ_SNAPSHOT_CHANGED: the journal source is no longer "
                "available; no mixed page was returned; restart with offset=0 and no snapshot."
            )
        return f"(journal for project {pid} is empty)"
    try:
        take = max(1, min(int(limit or 30), 200))
    except (TypeError, ValueError):
        take = 30
    try:
        skip = max(0, int(offset or 0))
    except (TypeError, ValueError):
        skip = 0
    rows, current_snapshot, stable, unreadable = _journal_snapshot_rows(path, pid)
    requested_snapshot = str(snapshot or "").strip().lower()
    if not stable:
        return (
            "JOURNAL_READ_SNAPSHOT_CHANGED_DURING_READ: the journal changed while "
            "the page was captured; no mixed page was returned; retry with offset=0 "
            "and no snapshot."
        )
    if requested_snapshot and requested_snapshot != current_snapshot:
        return (
            "JOURNAL_READ_SNAPSHOT_CHANGED: the journal changed after the prior page; "
            "no mixed page was returned; restart with offset=0 and no snapshot."
        )
    valid_total = len(rows)
    total = valid_total + unreadable
    end = max(0, valid_total - skip)
    start = max(0, end - take)
    page = rows[start:end]
    remaining = start
    lines = [
        f"Page: total={total} valid_total={valid_total} unreadable={unreadable} "
        f"returned={len(page)} offset={skip} remaining={remaining} "
        f"remaining_scope=valid_rows coverage={'partial' if unreadable else 'complete'} "
        f"snapshot={current_snapshot}"
    ]
    if unreadable:
        lines.append(
            f"JOURNAL_READ_GAP: coverage is partial; {unreadable} non-empty physical "
            "row(s) were malformed or non-object JSON. Valid rows remain pageable."
        )
    if remaining:
        lines.append(
            "Next older page: "
            f"journal_read(project_id='{pid}', limit={take}, "
            f"offset={skip + len(page)}, snapshot='{current_snapshot}')"
        )
    for row in page:
        lines.append(
            f"[{str(row.get('ts') or '')[:19]}] {str(row.get('kind') or 'note').upper()}: "
            f"{str(row.get('text') or '')}"
        )
    return f"## Project journal ({pid})\n\n" + "\n".join(lines)


def _workpad_read(ctx: ToolContext, project_id: str = "") -> str:
    pid, scope_error = _resolve_project_id(ctx, project_id, write=False)
    if scope_error:
        return scope_error + " (workpad_read)"
    if not pid:
        return "⚠️ TOOL_ARG_ERROR (workpad_read): no project scope."
    path = project_workpad_path(pid)
    if not path.is_file():
        return f"(workpad for project {pid} is empty)"
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        return f"⚠️ TOOL_ERROR (workpad_read): {exc}"


def _workpad_write(ctx: ToolContext, content: str, project_id: str = "") -> str:
    pid, scope_error = _resolve_project_id(ctx, project_id, write=True)
    if scope_error:
        return scope_error + " (workpad_write)"
    if not pid:
        return "⚠️ TOOL_ARG_ERROR (workpad_write): no project scope."
    body = str(content or "")
    if len(body.encode("utf-8", errors="ignore")) > _WORKPAD_MAX_BYTES:
        return ("⚠️ TOOL_ARG_ERROR (workpad_write): workpad exceeds 256KB — keep it a "
                "working page; move durable facts to knowledge_write and history to journal_write.")
    path = project_workpad_path(pid)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(body, encoding="utf-8")
    except OSError as exc:
        return f"⚠️ TOOL_ERROR (workpad_write): {exc}"
    return f"OK: workpad[{pid}] written ({len(body)} chars)."


_FOCUS_SOURCE_REFUSAL_PREFIXES = (
    "⚠️", "JOURNAL_READ_SNAPSHOT_CHANGED", "CHAT_HISTORY_SNAPSHOT_CHANGED",
)
# A focus points at ONE bounded page of an existing reader; a retained answer is
# a cognitive artifact, not a mirror of the store. Larger answers are refused
# with the repair (narrow the page), never clipped.
_FOCUS_SOURCE_MAX_BYTES = 256 * 1024


def _read_focus_source(ctx: ToolContext, source: Dict[str, Any]) -> Tuple[Optional[str], str]:
    """Answer the focus source_ref through the very reader it names, as the caller.

    Returns ``(text, "")`` when the reader answered, else ``("", reason)``.  The
    call runs under the caller's own authority (the same scope checks the
    reader applies to the model), so a focus cannot retain what its author
    could not read.  A typed refusal or a snapshot mismatch is NOT evidence:
    the focus is refused instead of pointing at nothing.
    """
    reader = str(source.get("reader") or "")
    args = {key: item for key, item in source.items() if key != "reader"}
    # The same admission the registry applies to a direct call of that reader:
    # a task whose contract withholds `journal_read` cannot read it through
    # update_focus either (and a consciousness Observe level keeps its argument
    # refusals).  Retention never widens what the caller could dispatch.
    try:
        from ouroboros.tools.registry_guards import _capability_resource_guard_result

        guard = _capability_resource_guard_result(ctx, reader, dict(args))
    except Exception as exc:  # a guard that cannot be consulted is not permission
        log.debug("focus source guard failed", exc_info=True)
        return "", f"{reader} admission could not be established ({type(exc).__name__})"
    if guard is not None:
        return "", f"{reader} withheld for this task: {str(getattr(guard, 'text', '') or '').splitlines()[0][:160]}"
    try:
        if reader == "journal_read":
            text = _journal_read(ctx, project_id=str(args.get("project_id") or ""),
                                 offset=int(args.get("offset") or 0), snapshot=str(args.get("snapshot") or ""))
        elif reader == "workpad_read":
            text = _workpad_read(ctx, project_id=str(args.get("project_id") or ""))
        elif reader == "recent_tasks":
            from ouroboros.tools.recent_tasks import _handle_recent_tasks
            text = _handle_recent_tasks(ctx, offset=int(args.get("offset") or 0), snapshot=str(args.get("snapshot") or ""))
        elif reader == "live_roots":
            from ouroboros.tools.recent_tasks import _handle_live_roots
            text = _handle_live_roots(ctx, offset=int(args.get("offset") or 0), snapshot=str(args.get("snapshot") or ""))
        elif reader == "get_task_result":
            from ouroboros.task_status import load_effective_task_result
            from ouroboros.task_status import SETTLED_STATUSES
            from ouroboros.routing_wait import is_emitted_admission_stub
            from ouroboros.tool_access import canonical_data_root
            from ouroboros.tools.control_task_results import _get_task_result
            row = load_effective_task_result(canonical_data_root(ctx), str(args.get("task_id") or ""))
            if not row or is_emitted_admission_stub(row) or str(row.get("status") or "") not in SETTLED_STATUSES:
                # The reader's unavailable/pending/running answers are prose
                # without a typed marker; a task with no settled result is not a
                # source (its live text is "Task is running.").
                return "", f"{reader} refused: task {args.get('task_id')} unknown, admission pending or not yet settled"
            text = _get_task_result(ctx, task_id=str(args.get("task_id") or ""))
        elif reader == "chat_history":
            from ouroboros.tools.control_runtime import _chat_history
            text = _chat_history(ctx, offset=int(args.get("offset") or 0), snapshot=str(args.get("snapshot") or ""))
        else:
            return "", f"reader {reader!r} has no resolver"
    except Exception as exc:  # a reader that raised answered nothing
        log.debug("focus source read failed", exc_info=True)
        return "", f"{reader} raised {type(exc).__name__}"
    body = str(text or "")
    stripped = body.lstrip()
    if not stripped:
        return "", f"{reader} answered nothing"
    if stripped.startswith(_FOCUS_SOURCE_REFUSAL_PREFIXES):
        return "", f"{reader} refused: {stripped.splitlines()[0][:160]}"
    if stripped.startswith("{"):
        try:
            payload = json.loads(stripped)
        except ValueError:
            payload = None
        if isinstance(payload, dict) and (payload.get("ok") is False or payload.get("error")):
            error = payload.get("error")
            code = error.get("code") if isinstance(error, dict) else payload.get("host_code")
            return "", f"{reader} refused: {code or 'typed error'}"
    if len(body.encode("utf-8")) > _FOCUS_SOURCE_MAX_BYTES:
        return "", (f"{reader} answered {len(body.encode('utf-8'))} bytes, above the {_FOCUS_SOURCE_MAX_BYTES}-byte "
                    "focus source bound; point the focus at a narrower page (offset/snapshot) or a smaller reader")
    return body, ""


def _retain_focus_source(canonical: pathlib.Path, task_id: str, source: Dict[str, Any], body: str) -> Dict[str, Any]:
    """Store the reader's exact answer write-once on the canonical root; return the handle.

    The handle is the native ``task_source`` ref minus its ``read`` block: peers
    resolve it through ``get_task_result(include_focus_source=True)`` against the
    same canonical root the durable task result lives on, not through the
    author's own (possibly forked) ``artifact_store``.
    """
    from ouroboros.artifacts import store_actor_source_bytes

    ref = store_actor_source_bytes(canonical, task_id, category="context_checkpoints",
                                   source_id=f"focus_source_{source.get('reader')}",
                                   data=body.encode("utf-8"), extension="md")
    return {key: ref[key] for key in ("kind", "root", "path", "size", "sha256")}


def _update_focus(ctx: ToolContext, text: str, source_ref: Any) -> str:
    """Publish one short authored focus onto this live root's existing records."""
    metadata = getattr(ctx, "task_metadata", {})
    metadata = metadata if isinstance(metadata, dict) else {}
    task_contract = getattr(ctx, "task_contract", {})
    task_contract = task_contract if isinstance(task_contract, dict) else {}
    task = {"metadata": metadata, "task_contract": task_contract,
            "_presence_turn": bool(getattr(ctx, "_presence_turn", False)),
            "_presence_origin": getattr(ctx, "_presence_origin", None)}
    authority, _ = _scope_authority(ctx)
    if authority != "root" or is_presence_task(task):
        return "⚠️ TOOL_FORBIDDEN (update_focus): only an independent project/root task may publish focus"
    task_id = str(getattr(ctx, "task_id", "") or "").strip()
    if not task_id:
        return "⚠️ TOOL_ARG_ERROR (update_focus): task_id is required"
    # The schema requires only ``reader``; a project reader without an explicit
    # project_id means the caller's own current project (the handler canonicalizes
    # what the provider was allowed to omit).
    if isinstance(source_ref, dict) and str(source_ref.get("reader") or "") in ("journal_read", "workpad_read") \
            and not str(source_ref.get("project_id") or "").strip():
        own_project = str(getattr(ctx, "project_id", "") or "").strip()
        if own_project:
            source_ref = {**source_ref, "project_id": own_project}
    from ouroboros.task_results import STATUS_RUNNING, load_task_result, write_task_result
    import pathlib

    canonical = pathlib.Path(str(metadata.get("budget_drive_root") or getattr(ctx, "budget_drive_root", "") or getattr(ctx, "drive_root", "")))
    direct = bool(getattr(ctx, "is_direct_chat", False))
    if direct:
        try:
            from supervisor.workers import direct_chat_turn
            if direct_chat_turn(task_id) is None:
                return "⚠️ FOCUS_TASK_NOT_LIVE (update_focus): the current direct turn is no longer live"
        except Exception:
            return "⚠️ FOCUS_TASK_NOT_LIVE (update_focus): the current direct turn is no longer live"
    else:
        current = load_task_result(canonical, task_id)
        if isinstance(current, dict) and str(current.get("status") or "") != STATUS_RUNNING:
            return "⚠️ FOCUS_TASK_NOT_LIVE (update_focus): the current task is no longer live"
    try:
        focus = normalize_focus(text, source_ref, task_id=task_id)
    except ValueError as exc:
        return f"⚠️ TOOL_ARG_ERROR (update_focus): {exc}"
    # A source_ref is a pointer; the pointer must identify evidence after this
    # task goes dormant (a workpad or live-root page is rewritten by then).  Read
    # it now through the named reader and retain the exact answer beside the
    # focus, so peers read what the author saw rather than what the page says
    # later.  A source that cannot be read is not a source.
    body, refusal = _read_focus_source(ctx, focus["source_ref"])
    if refusal:
        return f"⚠️ FOCUS_SOURCE_UNRESOLVED (update_focus): {refusal}"
    try:
        handle = _retain_focus_source(canonical, task_id, focus["source_ref"], body)
        focus = normalize_focus(text, source_ref, task_id=task_id,
                                authored_at=focus["authored_at"], source_handle=handle)
    except (OSError, ValueError) as exc:
        log.debug("focus source retention failed", exc_info=True)
        return f"⚠️ FOCUS_SOURCE_UNRETAINED (update_focus): the source answer could not be retained ({type(exc).__name__})"

    def _project(current: Dict[str, Any], fields: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if str(current.get("status") or "") != STATUS_RUNNING:
            return None
        prior = current.get("focus")
        if prior:
            from ouroboros.focus import compact_focus
            prior_focus = compact_focus(prior)
            if prior_focus and str(prior_focus.get("authored_at") or "") >= str(focus.get("authored_at") or ""):
                return None
        return fields

    # A direct turn is owned by the live ingress actor rather than the queue.
    # Hold its existing admission lock across the liveness check and the locked
    # result write, then require the same actor projection to accept the stamp.
    lock = getattr(ctx, "owner_message_admission_lock", None) if direct else None
    guard = lock if lock is not None else nullcontext()
    try:
        with guard:
            if direct:
                from supervisor.workers import direct_chat_turn, stamp_direct_chat_turn
                if direct_chat_turn(task_id) is None:
                    return "⚠️ FOCUS_TASK_NOT_LIVE (update_focus): the current direct turn is no longer live"
            current = load_task_result(canonical, task_id)
            if not isinstance(current, dict) or str(current.get("status") or "") != STATUS_RUNNING:
                return "⚠️ FOCUS_TASK_NOT_LIVE (update_focus): the current task is no longer live"
            stored = write_task_result(canonical, task_id, STATUS_RUNNING, focus=focus, _field_projector=_project)
            if not isinstance(stored, dict) or str(stored.get("status") or "") != STATUS_RUNNING:
                return "⚠️ FOCUS_TASK_NOT_LIVE (update_focus): the current task is no longer live"
            if stored.get("focus") != focus:
                return "⚠️ FOCUS_STALE (update_focus): a newer focus already exists"
            if direct:
                if not stamp_direct_chat_turn(task_id, focus=focus):
                    return "⚠️ FOCUS_PROJECTION_UNAVAILABLE (update_focus): the direct turn projection rejected the focus"
    except Exception:
        log.debug("focus persistence failed", exc_info=True)
        return "⚠️ TOOL_ERROR (update_focus): live focus could not be persisted"
    event_queue = getattr(ctx, "event_queue", None)
    if event_queue is not None:
        try:
            event_queue.put_nowait({"type": "task_focus_updated", "task_id": task_id, "focus": focus})
        except Exception:
            log.debug("focus projection event failed", exc_info=True)
            # The durable task-result write succeeded, but the fast projection
            # notification did not.  Do not claim a fully published update:
            # callers may retry, while peer_roster independently reconciles
            # the durable carrier on its next read.
            return ("⚠️ FOCUS_PROJECTION_UNAVAILABLE (update_focus): focus was stored, "
                    "but the live projection notification could not be queued")
    return f"OK: focus[{task_id}] updated (authored_at={focus['authored_at']})."


def journal_tail_digest(project_id: str, *, limit: int = 40) -> str:
    """Recent project-journal milestones for context injection (no ctx needed).

    Cognitive artifact (BIBLE P1): each milestone is shown in FULL — never
    per-row prefix-sliced. Older entries beyond the tail are represented by a
    VISIBLE index pointer to journal_read (horizon preserved via pointer,
    granularity varies), never silently dropped."""
    pid = sanitize_project_id(project_id)
    if not pid:
        return ""
    path = project_journal_path(pid)
    if not path.is_file():
        return ""
    rows, snapshot, stable, unreadable = _journal_snapshot_rows(path, pid)
    if not rows and not unreadable:
        return ""
    page_size = max(1, int(limit))
    take = rows[-page_size:]
    omitted = len(rows) - len(take)
    lines = [
        f"- [{str(r.get('ts') or '')[:16]}] {str(r.get('kind') or 'note')}: {str(r.get('text') or '')}"
        for r in take
    ]
    if unreadable:
        lines.insert(0, (
            f"- ⚠ coverage partial: {unreadable} unreadable journal row(s); "
            f"inspect valid rows with journal_read(project_id='{pid}')"
        ))
    if omitted:
        if stable:
            lines.insert(0, (
                f"- …[{omitted} earlier milestones; source="
                f"journal_read(project_id='{pid}'); next="
                f"journal_read(project_id='{pid}', limit={page_size}, "
                f"offset={len(take)}, snapshot='{snapshot}') ]"
            ))
        else:
            lines.insert(0, (
                f"- …[{omitted} observed earlier milestones; journal changed during "
                f"capture; restart with journal_read(project_id='{pid}', limit={page_size})]"
            ))
    return "\n".join(lines)


def get_tools() -> List[ToolEntry]:
    common = {
        "project_id": {
            "type": "string",
            "description": "Explicit project id (defaults to the current task's project scope).",
            "default": "",
        },
    }
    return [
        ToolEntry(
            "update_focus",
            {
                "name": "update_focus",
                "description": (
                    "Publish a short authored focus for this live root. The source_ref is a "
                    "typed reader/cursor reference; focus is awareness, never an owner directive."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "Short focus text (<=280 chars)."},
                        "source_ref": {
                            "type": "object",
                            "description": "Reference to an existing bounded reader; this carries no content or path.",
                            "properties": {
                                "reader": {"type": "string", "enum": [
                                    "journal_read", "workpad_read", "recent_tasks",
                                    "get_task_result", "chat_history", "live_roots",
                                ]},
                                "project_id": {"type": "string"},
                                "task_id": {"type": "string"},
                                "offset": {"type": "integer", "minimum": 0},
                                "snapshot": {"type": "string"},
                            },
                            "required": ["reader"],
                            "additionalProperties": False,
                        },
                    },
                    "required": ["text", "source_ref"],
                },
            },
            lambda ctx, text, source_ref: _update_focus(ctx, text, source_ref),
            timeout_sec=15,
        ),
        ToolEntry(
            "journal_write",
            {
                "name": "journal_write",
                "description": (
                    "Append a milestone entry to the current project's durable journal. "
                    "kind: start | checkpoint | blocked | done | note. The journal is the "
                    "project's long-term progress memory (survives task restarts; feeds "
                    "the owner-visible project digest)."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "kind": {"type": "string", "enum": list(_JOURNAL_KINDS)},
                        "text": {"type": "string", "description": "Milestone text (<=4000 chars)."},
                        **common,
                    },
                    "required": ["kind", "text"],
                },
            },
            lambda ctx, kind, text, project_id="": _journal_write(ctx, kind, text, project_id),
            timeout_sec=15,
        ),
        ToolEntry(
            "journal_read",
            {
                "name": "journal_read",
                "description": "Read the tail of the current project's journal (newest last).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "limit": {"type": "integer", "default": 30, "description": "Max entries (<=200)."},
                        "offset": {
                            "type": "integer",
                            "default": 0,
                            "description": "Number of newer entries already consumed.",
                        },
                        "snapshot": {
                            "type": "string",
                            "default": "",
                            "description": "Stable cursor returned by the preceding page.",
                        },
                        **common,
                    },
                },
            },
            lambda ctx, limit=30, offset=0, snapshot="", project_id="": _journal_read(
                ctx, project_id, limit, offset, snapshot,
            ),
            timeout_sec=15,
        ),
        ToolEntry(
            "workpad_read",
            {
                "name": "workpad_read",
                "description": "Read the current project's free-form workpad page.",
                "parameters": {"type": "object", "properties": dict(common)},
            },
            lambda ctx, project_id="": _workpad_read(ctx, project_id),
            timeout_sec=15,
        ),
        ToolEntry(
            "workpad_write",
            {
                "name": "workpad_write",
                "description": (
                    "Overwrite the current project's workpad page (<=256KB). A working "
                    "page for plans/links/state — durable facts belong in knowledge_write, "
                    "history in journal_write."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string"},
                        **common,
                    },
                    "required": ["content"],
                },
            },
            lambda ctx, content, project_id="": _workpad_write(ctx, content, project_id),
            timeout_sec=15,
        ),
    ]


__all__ = ["get_tools", "journal_tail_digest"]

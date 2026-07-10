"""Durable registry of owner projects (multi-project, v6.32.0).

A project is a durable context the single agent works in: id + name +
per-project memory (``data/projects/<id>/``) + chat thread (its own positive
``chat_id``) + an OPTIONAL working folder (invisible auto-git under the
durable projects root). File-less research projects are valid. Projects are
NEVER age-pruned; the owner curates by archive/delete.

State lives in ``data/state/projects.json`` via the canonical durable-JSON
pattern (mirrors ``subagent_worktrees.py``). The registry is data-plane
bookkeeping only — identity, constitution, and evolution stay unified in the
one agent (BIBLE P1).
"""

from __future__ import annotations

import logging
import pathlib
import threading
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional

from ouroboros.contracts.chat_id_policy import project_chat_id
from ouroboros.contracts.schema_versions import with_schema_version
from ouroboros.project_facts import sanitize_project_id
from ouroboros.utils import atomic_write_json, iter_jsonl_objects, read_json_dict, utc_now_iso

log = logging.getLogger(__name__)

_REGISTRY_NAME = "projects.json"
_BINDINGS_NAME = "project_task_bindings.json"
# v6.58.0 (slice 0): projects.json carries an opt-in _schema_version so future
# additive fields (git provenance, trusted_at) migrate deliberately. Old rows read
# as version 0; new fields must stay additive with safe-empty defaults because
# reconcile_projects mints rows that will lack them.
_REGISTRY_SCHEMA_VERSION = 1
_LOCK = threading.Lock()


@contextmanager
def _file_write_lock(target_path: pathlib.Path) -> Iterator[None]:
    """Cross-process exclusive lock for a registry/bindings read-modify-write.

    The registry is written from BOTH the server process (project create/bind,
    digest touch) AND worker processes (``project_journal`` touch_project), so a
    process-local ``threading.Lock`` cannot prevent lost updates. Flock a sidecar
    so the load→modify→atomic-write sequence is exclusive across processes; the
    in-process ``_LOCK`` is nested inside for thread-level serialization too.
    """
    from ouroboros.platform_layer import (
        acquire_exclusive_file_lock,
        release_exclusive_file_lock,
    )

    target_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = target_path.with_name(target_path.name + ".lock")
    fd = acquire_exclusive_file_lock(lock_path, timeout_sec=4.0)
    if fd is None:
        raise TimeoutError(f"projects_registry: could not lock {lock_path} in time")
    try:
        with _LOCK:
            yield
    finally:
        release_exclusive_file_lock(lock_path, fd)


def _registry_path(drive_root: Any) -> pathlib.Path:
    return pathlib.Path(drive_root) / "state" / _REGISTRY_NAME


def _bindings_path(drive_root: Any) -> pathlib.Path:
    return pathlib.Path(drive_root) / "state" / _BINDINGS_NAME


def _load(drive_root: Any) -> Dict[str, Any]:
    data = read_json_dict(_registry_path(drive_root))
    if not isinstance(data, dict) or not isinstance(data.get("projects"), list):
        return {"projects": []}
    data["projects"] = [p for p in data["projects"] if isinstance(p, dict) and p.get("id")]
    return data


def _save(drive_root: Any, data: Dict[str, Any]) -> None:
    path = _registry_path(drive_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Stamp the current schema version on every write (idempotent; old files that
    # never had it are treated as version 0 by read_schema_version).
    atomic_write_json(path, with_schema_version(dict(data), _REGISTRY_SCHEMA_VERSION))


def _load_bindings(drive_root: Any) -> Dict[str, Any]:
    data = read_json_dict(_bindings_path(drive_root))
    if not isinstance(data, dict) or not isinstance(data.get("bindings"), dict):
        return {"bindings": {}}
    return data


def _save_bindings(drive_root: Any, data: Dict[str, Any]) -> None:
    path = _bindings_path(drive_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, data)


def bind_task_to_project(drive_root: Any, task_id: str, project_id: str, chat_id: Any = None) -> Dict[str, Any]:
    """Durably bind an existing task/live card to a project thread.

    This is the post-hoc "Turn into project" bridge: old audit logs remain in
    their original files, while history/live routing can resolve the task's
    project chat from this lightweight binding.
    """
    tid = str(task_id or "").strip()
    pid = sanitize_project_id(project_id)
    if not tid:
        raise ValueError("task_id is required")
    if not pid:
        raise ValueError(f"unusable project id: {project_id!r}")
    project = get_project(drive_root, pid) or create_project(drive_root, pid)
    try:
        resolved_chat = int(chat_id if chat_id is not None else project.get("chat_id"))
    except (TypeError, ValueError):
        resolved_chat = project_chat_id(pid)
    row = {
        "task_id": tid,
        "project_id": pid,
        "project_chat_id": resolved_chat,
        "bound_at": utc_now_iso(),
    }
    with _file_write_lock(_bindings_path(drive_root)):
        data = _load_bindings(drive_root)
        data["bindings"][tid] = row
        _save_bindings(drive_root, data)
    touch_project(drive_root, pid)
    return dict(row)


def all_task_bindings(drive_root: Any) -> Dict[str, int]:
    """Map task_id -> project chat_id for ALL post-hoc 'Turn into project' bindings.

    Cognition/history isolation consults this so a bound task's rows (which keep
    their ORIGINAL main chat_id) are still treated as project-owned. One bounded
    read; no per-row lock (atomic writes guarantee complete reads)."""
    out: Dict[str, int] = {}
    try:
        for tid, row in _load_bindings(drive_root).get("bindings", {}).items():
            if not isinstance(row, dict):
                continue
            try:
                cid = int(row.get("project_chat_id") or 0)
            except (TypeError, ValueError):
                continue
            if cid:
                out[str(tid)] = cid
    except Exception:
        log.debug("all_task_bindings failed", exc_info=True)
    return out


def all_task_project_bindings(drive_root: Any) -> Dict[str, Dict[str, Any]]:
    """Map task_id -> {project_id, chat_id} for ALL post-hoc 'Turn into project'
    bindings. Richer than all_task_bindings (chat-id only): the UI uses project_id
    to turn a bound main-chat card into a pointer that opens the project panel
    (F4), not merely to suppress the stray convert button (P2). Never raises."""
    out: Dict[str, Dict[str, Any]] = {}
    try:
        for tid, row in _load_bindings(drive_root).get("bindings", {}).items():
            if not isinstance(row, dict):
                continue
            pid = str(row.get("project_id") or "").strip()
            try:
                cid = int(row.get("project_chat_id") or 0)
            except (TypeError, ValueError):
                cid = 0
            if pid and cid:
                out[str(tid)] = {"project_id": pid, "chat_id": cid}
    except Exception:
        log.debug("all_task_project_bindings failed", exc_info=True)
    return out


def project_binding_for_task(drive_root: Any, task_id: str) -> Optional[Dict[str, Any]]:
    tid = str(task_id or "").strip()
    if not tid:
        return None
    # Read needs no lock: atomic_write_json renames into place, so a reader
    # always sees a complete (old or new) bindings file, never a torn one.
    row = _load_bindings(drive_root)["bindings"].get(tid)
    return dict(row) if isinstance(row, dict) else None


def project_chat_for_task(drive_root: Any, task_id: str) -> int:
    row = project_binding_for_task(drive_root, task_id)
    if not row:
        return 0
    try:
        return int(row.get("project_chat_id") or 0)
    except (TypeError, ValueError):
        return 0


def project_chat_for_task_tree(
    drive_root: Any, task_id: Any, parent_task_id: Any = "", root_task_id: Any = ""
) -> int:
    """Resolve the project chat for a task by its TASK TREE: the task's OWN binding
    wins; else inherit from its parent; else its root. A subagent is never bound
    itself, so this is how its live frames + history are recognized as belonging to
    its root's project and route to the project thread instead of staying in the main
    chat (the cyber-racing "subagents vanished from the project" gap). Membership is
    DERIVED from lineage — no per-child binding store, one SSOT."""
    for tid in (task_id, parent_task_id, root_task_id):
        tid = str(tid or "").strip()
        if not tid:
            continue
        chat = project_chat_for_task(drive_root, tid)
        if chat:
            return chat
    return 0


def list_projects(drive_root: Any) -> List[Dict[str, Any]]:
    """All registered projects (most recently active first)."""
    with _LOCK:
        projects = _load(drive_root)["projects"]
    return sorted(
        projects,
        key=lambda p: str(p.get("last_active_at") or p.get("updated_at") or p.get("created_at") or ""),
        reverse=True,
    )


def registered_project_chat_ids(drive_root: Any) -> set:
    """The set of chat_ids owned by ALL registered projects.

    The TRUTH source for "is this chat a project thread" — a bare numeric range
    cannot disambiguate from large external-transport (e.g. Telegram) chat ids,
    so routing/history/UI classify by registry membership instead.

    NOT an isolation boundary (full project awareness, v6.32.0): the one identity
    sees ALL threads in its unified memory. This classifier drives (a) the UI
    history/fan-out partition that organizes threads into panels, (b) message
    routing, and (c) the project TASK's FOCUSED passive context (build_recent_
    sections shows the task its own thread).
    """
    out = set()
    try:
        for project in list_projects(drive_root):
            try:
                out.add(int(project.get("chat_id") or 0))
            except (TypeError, ValueError):
                continue
    except Exception:
        log.debug("registered_project_chat_ids failed", exc_info=True)
    out.discard(0)
    return out


def get_project(drive_root: Any, project_id: str) -> Optional[Dict[str, Any]]:
    pid = sanitize_project_id(project_id)
    if not pid:
        return None
    for project in list_projects(drive_root):
        if project.get("id") == pid:
            return dict(project)
    return None


def create_project(
    drive_root: Any,
    project_id: str,
    *,
    name: str = "",
    working_dir: str = "",
    origin: str = "owner",
) -> Dict[str, Any]:
    """Register (or idempotently return) a project entry.

    ``working_dir`` is optional — file-less projects (research, presentations
    drafted in chat) are first-class. The per-project chat id is derived
    deterministically from the id (one allocator-free SSOT).
    """
    pid = sanitize_project_id(project_id)
    if not pid:
        raise ValueError(f"unusable project id: {project_id!r}")
    with _file_write_lock(_registry_path(drive_root)):
        data = _load(drive_root)
        for existing in data["projects"]:
            if existing.get("id") == pid:
                return dict(existing)
        entry = {
            "id": pid,
            "name": str(name or "").strip() or pid,
            "chat_id": project_chat_id(pid),
            "working_dir": str(working_dir or "").strip(),
            "origin": str(origin or "owner"),
            "created_at": utc_now_iso(),
            "last_active_at": utc_now_iso(),
        }
        data["projects"].append(entry)
        _save(drive_root, data)
        log.info("Project registered: %s (chat_id=%s)", pid, entry["chat_id"])
        return dict(entry)


def update_project(drive_root: Any, project_id: str, **updates: Any) -> Optional[Dict[str, Any]]:
    """Update mutable fields. v6.59.0 adds the additive source-provenance facts:
    ``provenance`` (attached|cloned|genesis|none — how the working_dir came to be),
    ``clone_url`` (historical fact; live git data is always read from .git), and
    ``trusted_at`` (stamped automatically on attach/clone — the notification trust
    model: attaching IS the owner's explicit grant, no second confirmation gate)."""
    pid = sanitize_project_id(project_id)
    if not pid:
        return None
    allowed = {"name", "working_dir", "last_active_at", "provenance", "clone_url", "trusted_at"}
    with _file_write_lock(_registry_path(drive_root)):
        data = _load(drive_root)
        for entry in data["projects"]:
            if entry.get("id") != pid:
                continue
            for key, value in updates.items():
                if key not in allowed:
                    continue
                entry[key] = value
            _save(drive_root, data)
            return dict(entry)
    return None


def delete_project(drive_root: Any, project_id: str) -> bool:
    """v6.59.0: remove a project's REGISTRATION + its task bindings. The working
    folder and the per-project memory store are deliberately NOT touched — delete
    un-registers, it never destroys owner data (the folder belongs to the owner;
    the memory store is durable and reconcile could resurrect the row if wanted).
    Returns True when a row was removed."""
    pid = sanitize_project_id(project_id)
    if not pid:
        return False
    removed = False
    with _file_write_lock(_registry_path(drive_root)):
        data = _load(drive_root)
        kept = [p for p in data["projects"] if p.get("id") != pid]
        if len(kept) != len(data["projects"]):
            data["projects"] = kept
            _save(drive_root, data)
            removed = True
    if removed:
        with _file_write_lock(_bindings_path(drive_root)):
            bindings = _load_bindings(drive_root)
            pruned = {
                tid: row for tid, row in bindings.get("bindings", {}).items()
                if not (isinstance(row, dict) and str(row.get("project_id") or "") == pid)
            }
            if len(pruned) != len(bindings.get("bindings", {})):
                bindings["bindings"] = pruned
                _save_bindings(drive_root, bindings)
        log.info("Project unregistered: %s (folder and memory store untouched)", pid)
    return removed


def touch_project(drive_root: Any, project_id: str) -> None:
    """Record activity (never raises)."""
    try:
        update_project(drive_root, project_id, last_active_at=utc_now_iso())
    except Exception:
        log.debug("touch_project failed for %s", project_id, exc_info=True)


def reconcile_projects(drive_root: Any) -> int:
    """Boot reconcile: register projects whose memory store exists but whose
    registry row is missing (e.g. created before the registry shipped, or a
    workspace-derived ``proj_<hash>`` store). NEVER prunes — durable project
    dirs outlive any registry accident.
    """
    added = 0
    try:
        projects_root = pathlib.Path(drive_root) / "projects"
        if not projects_root.is_dir():
            return 0
        with _file_write_lock(_registry_path(drive_root)):
            data = _load(drive_root)
            known = {p.get("id") for p in data["projects"]}
            for entry in sorted(projects_root.iterdir()):
                if not entry.is_dir() or entry.name.startswith("."):
                    continue
                pid = sanitize_project_id(entry.name)
                if not pid or pid in known:
                    continue
                data["projects"].append({
                    "id": pid,
                    "name": pid,
                    "chat_id": project_chat_id(pid),
                    "working_dir": "",
                    "origin": "reconcile",
                    "created_at": utc_now_iso(),
                    "last_active_at": utc_now_iso(),
                })
                known.add(pid)
                added += 1
            if added:
                _save(drive_root, data)
                log.info("Project registry reconcile: %d store(s) registered", added)
    except Exception:
        log.warning("Project registry reconcile failed", exc_info=True)
    return added


def ensure_project_workspace(drive_root: Any, project_id: str, repo_dir: Any) -> str:
    """Provision (once) an invisible-git working folder for a project.

    Reuses the genesis-project machinery: a standalone git repo under the
    durable projects root (never GC-pruned, isolated from repo/ and data/).
    Returns the absolute path ("" when provisioning failed). File-less
    projects simply never call this.
    """
    entry = get_project(drive_root, project_id)
    if entry is None:
        entry = create_project(drive_root, project_id)
    existing = str(entry.get("working_dir") or "").strip()
    if existing and pathlib.Path(existing).is_dir():
        return existing
    try:
        from ouroboros.subagent_worktrees import provision_genesis_project

        handle = provision_genesis_project(
            repo_dir=repo_dir,
            task_id=f"project_{entry['id']}",
            data_dir=drive_root,
            # Name the genesis folder after the project so sibling builders land in a
            # recognizable shared root (binding identity stays the task_id). (I, v6.39)
            dir_name=str(entry.get("name") or ""),
        )
        update_project(drive_root, entry["id"], working_dir=str(handle.path))
        return str(handle.path)
    except Exception:
        log.warning("Project workspace provisioning failed for %s", project_id, exc_info=True)
        return ""


def projects_summary(drive_root: Any, *, limit: int = 50) -> List[Dict[str, Any]]:
    """Compact list for /api/state and the sidebar."""
    out: List[Dict[str, Any]] = []
    bindings = _load_bindings(drive_root).get("bindings", {})

    def _has_thread_activity(project: Dict[str, Any]) -> bool:
        pid = str(project.get("id") or "")
        # v6.59.0: a project the OWNER explicitly created in the UI is always shown —
        # the activity filter exists to hide junk reconcile rows, not a fresh project
        # the owner just made (which has no chat rows yet by definition).
        if str(project.get("origin") or "") == "owner_ui":
            return True
        try:
            cid = int(project.get("chat_id") or 0)
        except (TypeError, ValueError):
            cid = 0
        if any(isinstance(row, dict) and row.get("project_id") == pid for row in bindings.values()):
            return True
        if not cid:
            return False
        logs = pathlib.Path(drive_root) / "logs"
        for rel in ("chat.jsonl", "progress.jsonl"):
            path = logs / rel
            if not path.is_file():
                continue
            try:
                for row in iter_jsonl_objects(path):
                    try:
                        if int(row.get("chat_id") or 1) == cid:
                            return True
                    except (TypeError, ValueError):
                        continue
            except Exception:
                log.debug("project activity scan failed for %s", path, exc_info=True)
        return False

    for project in list_projects(drive_root)[: max(1, int(limit))]:
        out.append({
            "id": project.get("id"),
            "name": project.get("name"),
            "chat_id": project.get("chat_id"),
            "working_dir": project.get("working_dir") or "",
            "provenance": project.get("provenance") or "",
            "last_active_at": project.get("last_active_at") or "",
            "has_thread_activity": _has_thread_activity(project),
        })
    return out


__all__ = [
    "all_task_bindings",
    "bind_task_to_project",
    "create_project",
    "delete_project",
    "ensure_project_workspace",
    "get_project",
    "list_projects",
    "project_binding_for_task",
    "project_chat_for_task",
    "project_chat_for_task_tree",
    "registered_project_chat_ids",
    "projects_summary",
    "reconcile_projects",
    "touch_project",
    "update_project",
]

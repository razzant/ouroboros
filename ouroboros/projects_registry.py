"""Durable registry of owner projects (multi-project, v6.32.0).

A project is a durable context the single agent works in: id + name +
per-project memory (``data/projects/<id>/``) + chat thread (its own positive
``chat_id``) + an OPTIONAL working folder (invisible auto-git under the
durable projects root). File-less research projects are valid. Projects are
NEVER age-pruned; the owner curates by archive/delete.

State lives in ``data/state/projects.json`` via the canonical durable-JSON
pattern (mirrors ``subagent_worktrees.py``). Deletion keeps a durable tombstone
so chat history, bindings, memory and the owner folder remain addressable and a
boot reconcile cannot resurrect the room. The registry is data-plane
bookkeeping only — identity, constitution, and evolution stay unified in the
one agent (BIBLE P1).
"""

from __future__ import annotations

import logging
import pathlib
import threading
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional

from ouroboros.contracts.chat_id_policy import MAIN_THREAD_ID, project_chat_id
from ouroboros.contracts.schema_versions import with_schema_version
from ouroboros.project_facts import sanitize_project_id

# The thread block lives in its own module (module-size gate) but stays part of
# THIS module's public surface: every name below is importable from
# ``ouroboros.projects_registry`` exactly as before, so no caller moved. The
# borrow is one-way — project_threads_registry imports the registry's locking/IO
# primitives inside the functions that need them, never at module level.
from ouroboros.project_threads_registry import (
    THREAD_ACTIVE,
    THREAD_ARCHIVED,
    THREAD_DELETING,
    THREAD_TOMBSTONED,
    ThreadLifecycleError,
    _chat_id_owners,
    _normalize_thread_rows,
    _report_duplicate_chat_ids,
    _row_chat_ids,
    archive_thread,
    begin_thread_deletion,
    complete_thread_deletion,
    create_thread,
    duplicate_chat_ids,
    fail_thread_deletion,
    fork_thread,
    get_thread,
    project_thread_note_for_task,
    project_threads,
    rename_thread,
    restore_thread,
    thread_is_visible,
)
from ouroboros.utils import (
    atomic_write_json, iter_jsonl_objects, read_json_dict, truncate_review_artifact, utc_now_iso,
)

log = logging.getLogger(__name__)

_REGISTRY_NAME = "projects.json"
_BINDINGS_NAME = "project_task_bindings.json"
# v6.58.0 (slice 0): projects.json carries an opt-in _schema_version so future
# additive fields (git provenance, trusted_at) migrate deliberately. Old rows read
# as version 0; new fields must stay additive with safe-empty defaults because
# reconcile_projects mints rows that will lack them.
_REGISTRY_SCHEMA_VERSION = 2
# v6.73.0: project_task_bindings.json gains source_text / origin_absent fields.
_BINDINGS_SCHEMA_VERSION = 1
_LOCK = threading.RLock()

PROJECT_NAME_MAX = 80
PROJECT_ACTIVE = "active"
PROJECT_DELETING = "deleting"
PROJECT_TOMBSTONED = "tombstoned"

#: Character bound for the durable ``delete_error`` disclosure. A BOUND, not a
#: silent cut: overflow rides the ``truncate_review_artifact`` omission marker.
DELETE_ERROR_LIMIT = 4000
PROJECT_LIFECYCLES = frozenset({PROJECT_ACTIVE, PROJECT_DELETING, PROJECT_TOMBSTONED})
_DEPRECATED_CHAT_IDS_EVENTS: set[str] = set()

# Threads (project-threads T0). A project row carries an ADDITIVE ``threads: []``
# list of EXTRA threads; thread #0 is never stored — it is projected at read time
# from the project's own chat_id/name/timestamps/revision, and the top-level
# ``chat_id`` stays its compatibility alias. Nothing is rewritten on disk, so a
# legacy row (and any row minted by reconcile) reads as a one-thread project.
# The thread members themselves live in ``project_threads_registry``; the bound
# stays HERE because it is the project name bound (one rule, one constant) and
# reading it from there would need an import-time borrow in the wrong direction.
THREAD_NAME_MAX = PROJECT_NAME_MAX


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
    data["projects"] = [
        _normalize_project_row(p)
        for p in data["projects"]
        if isinstance(p, dict) and p.get("id")
    ]
    _report_duplicate_chat_ids(drive_root, data["projects"])
    return data


def _normalize_project_row(value: Dict[str, Any]) -> Dict[str, Any]:
    """Add safe lifecycle/read-cursor defaults without rewriting on read."""
    row = dict(value)
    lifecycle = str(row.get("lifecycle") or PROJECT_ACTIVE).strip().lower()
    row["lifecycle"] = lifecycle if lifecycle in PROJECT_LIFECYCLES else PROJECT_ACTIVE
    for field in ("routing_generation", "visible_revision"):
        try:
            row[field] = max(0, int(row.get(field) or 0))
        except (TypeError, ValueError):
            row[field] = 0
    row["delete_error"] = str(row.get("delete_error") or "")
    # Thread #0 owns its OWN revision counter; `visible_revision` stays the
    # project-wide AGGREGATE the flat `project_seen_revision` cursor compares
    # against. Sharing one number made every extra thread's activity mark thread
    # #0 unread as well. A legacy row seeds thread #0 from the aggregate: while a
    # project had exactly one thread the two numbers WERE the same fact, so the
    # seed is exact and no unread state is invented or lost.
    if "thread0_visible_revision" not in row:
        row["thread0_visible_revision"] = row["visible_revision"]
    else:
        try:
            row["thread0_visible_revision"] = max(
                0, int(row.get("thread0_visible_revision") or 0)
            )
        except (TypeError, ValueError):
            row["thread0_visible_revision"] = 0
    row["threads"] = _normalize_thread_rows(row.get("threads"))
    return row


def _validated_name(value: Any, fallback: str = "") -> str:
    name = str(value or "").strip() or str(fallback or "").strip()
    if len(name) > PROJECT_NAME_MAX:
        raise ValueError(f"project name must be <= {PROJECT_NAME_MAX} characters")
    return name


def _save(drive_root: Any, data: Dict[str, Any]) -> None:
    path = _registry_path(drive_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Stamp the current schema version on every write (idempotent; old files that
    # never had it are treated as version 0 by read_schema_version).
    atomic_write_json(path, with_schema_version(dict(data), _REGISTRY_SCHEMA_VERSION))
    # Monotonic in-process write counter feeding the chat-binding index memo:
    # a same-size rewrite inside one mtime tick is otherwise indistinguishable
    # from no write at all (see _CHAT_BINDING_INDEX).
    key = str(path)
    _REGISTRY_WRITE_SEQ[key] = _REGISTRY_WRITE_SEQ.get(key, 0) + 1


def _load_bindings(drive_root: Any) -> Dict[str, Any]:
    data = read_json_dict(_bindings_path(drive_root))
    if not isinstance(data, dict) or not isinstance(data.get("bindings"), dict):
        return {"bindings": {}}
    return data


def _save_bindings(drive_root: Any, data: Dict[str, Any]) -> None:
    path = _bindings_path(drive_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    # v6.73.0: bindings carry an opt-in _schema_version (legacy files read as 0).
    atomic_write_json(path, with_schema_version(dict(data), _BINDINGS_SCHEMA_VERSION))


# Closed enum of typed origin-absence reasons. ``producer_missing_ref`` is the
# truthful signal for a chat-born event whose producer failed to attach the ref
# (a grep-able producer bug, never a silent default). Upgrade-window note: tasks
# QUEUED before v6.73.0 predate the ingress capture, so their post-upgrade
# promotes legitimately land here until the pre-upgrade queue drains.
# NB: headless tasks are project-SCOPED but never project-BOUND (benchmark
# constraint), so the enum deliberately has no 'headless' member — it stays an
# honest map of reasons that actually have producers.
ORIGIN_ABSENT_REASONS = frozenset({
    "system",
    "mid_task_no_origin",
    "post_hoc_unresolved",
    "producer_missing_ref",
})

_ORIGIN_REF_KEYS = ("chat_id", "client_message_id", "ts", "text_sha256")


def _validated_origin(origin: Any, resolved_chat: int) -> Dict[str, Any]:
    """Validate the REQUIRED typed origin of a binding; raise ValueError otherwise.

    Content-derived identity lookups are forbidden (DEVELOPMENT.md anti-pattern):
    the caller must pass the origin ref BY VALUE (captured at chat ingress) or a
    typed absence reason. ``text`` is required exactly when the origin lives in a
    DIFFERENT chat than the project room (cross-thread projection needs the
    retention-proof copy); a same-room origin renders natively and stores no copy.
    """
    if not isinstance(origin, dict) or ("ref" in origin) == ("absent" in origin):
        raise ValueError(
            "bind_task_to_project requires origin={'ref': {...}, 'text': ...} for a "
            "chat-born binding or origin={'absent': <reason>} — exactly one of 'ref'/'absent'"
        )
    if "absent" in origin:
        reason = str(origin.get("absent") or "")
        if reason not in ORIGIN_ABSENT_REASONS:
            raise ValueError(
                f"invalid origin absence reason {reason!r}; expected one of {sorted(ORIGIN_ABSENT_REASONS)}"
            )
        return {"origin_absent": reason}
    ref = origin.get("ref")
    if not isinstance(ref, dict) or any(ref.get(key) in (None, "") for key in _ORIGIN_REF_KEYS):
        raise ValueError(f"origin['ref'] must carry non-empty {_ORIGIN_REF_KEYS}")
    clean_ref = {key: ref.get(key) for key in _ORIGIN_REF_KEYS}
    try:
        cross_thread = int(clean_ref.get("chat_id") or 0) != int(resolved_chat or 0)
    except (TypeError, ValueError):
        cross_thread = True
    text = origin.get("text")
    if not cross_thread:
        return {"source_ref": clean_ref}
    if not isinstance(text, str) or not text.strip():
        raise ValueError(
            "origin['text'] (the full original message) is required for a cross-thread "
            "origin — it is the retention-proof copy the Project lens projects"
        )
    from ouroboros.project_dialogue import _text_sha256

    if _text_sha256(text) != str(clean_ref.get("text_sha256") or ""):
        raise ValueError("origin['text'] does not match origin['ref']['text_sha256'] (integrity check)")
    return {"source_ref": clean_ref, "source_text": text}


def bind_task_to_project(
    drive_root: Any,
    task_id: str,
    project_id: str,
    chat_id: Any = None,
    *,
    origin: Dict[str, Any],
) -> Dict[str, Any]:
    """Durably bind an existing task/live card to a project thread.

    This is the post-hoc "Turn into project" bridge: old audit logs remain in
    their original files, while history/live routing can resolve the task's
    project chat from this lightweight binding.

    ``origin`` is REQUIRED and typed (see ``_validated_origin``): either the
    owner-message ref captured at chat ingress (+full text for a cross-thread
    origin) or a closed-enum absence reason. A same-task same-project re-bind
    that supplies a valid ref UPGRADES a ref-less existing row (one-way
    enrichment); an existing valid ref is never changed.
    """
    tid = str(task_id or "").strip()
    pid = sanitize_project_id(project_id)
    if not tid:
        raise ValueError("task_id is required")
    if not pid:
        raise ValueError(f"unusable project id: {project_id!r}")
    if get_reserved_project(drive_root, pid) is None:
        create_project(drive_root, pid)
    # Linearize admission with the lifecycle fence. Holding the registry lock
    # through the short bindings append means begin_project_deletion either lands
    # before this bind (which is refused) or after it (which cancellation sees).
    with _file_write_lock(_registry_path(drive_root)):
        project = next(
            (row for row in _load(drive_root)["projects"] if row.get("id") == pid),
            None,
        )
        if not isinstance(project, dict) or project.get("lifecycle") != PROJECT_ACTIVE:
            lifecycle = project.get("lifecycle") if isinstance(project, dict) else "missing"
            raise ValueError(f"project {pid!r} is {lifecycle}; it cannot accept bindings")
        try:
            resolved_chat = int(chat_id if chat_id is not None else project.get("chat_id"))
        except (TypeError, ValueError):
            resolved_chat = project_chat_id(pid)
        origin_fields = _validated_origin(origin, resolved_chat)
        row = {
            "task_id": tid,
            "project_id": pid,
            "project_chat_id": resolved_chat,
            "bound_at": utc_now_iso(),
            **origin_fields,
        }
        with _file_write_lock(_bindings_path(drive_root)):
            data = _load_bindings(drive_root)
            existing = data["bindings"].get(tid)
            if isinstance(existing, dict):
                existing_pid = str(existing.get("project_id") or "")
                if existing_pid == pid:
                    # One-way enrichment: fill a ref-less row when a valid ref
                    # arrives; a stored valid ref is immutable (never replaced).
                    if not isinstance(existing.get("source_ref"), dict) and "source_ref" in origin_fields:
                        enriched = {
                            key: value for key, value in existing.items() if key != "origin_absent"
                        }
                        enriched.update(origin_fields)
                        data["bindings"][tid] = enriched
                        _save_bindings(drive_root, data)
                        return dict(enriched)
                    return dict(existing)
                raise ValueError(
                    f"task {tid!r} is already bound to project {existing_pid!r}; "
                    "project binding is immutable"
                )
            data["bindings"][tid] = row
            _save_bindings(drive_root, data)
    touch_project(drive_root, pid)
    return dict(row)


def project_task_bindings(drive_root: Any) -> Dict[str, Dict[str, Any]]:
    """Copy of the immutable task-to-Project bindings for read models."""
    return {
        str(task_id): dict(row)
        for task_id, row in _load_bindings(drive_root).get("bindings", {}).items()
        if isinstance(row, dict)
    }


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


def list_reserved_projects(drive_root: Any) -> List[Dict[str, Any]]:
    """All Project ids, including deleting/tombstoned history reservations."""
    with _LOCK:
        projects = _load(drive_root)["projects"]
    return sorted(
        projects,
        key=lambda p: str(p.get("last_active_at") or p.get("updated_at") or p.get("created_at") or ""),
        reverse=True,
    )


def list_projects(drive_root: Any) -> List[Dict[str, Any]]:
    """Active, routable Projects (most recently active first)."""
    return [
        project for project in list_reserved_projects(drive_root)
        if project.get("lifecycle") == PROJECT_ACTIVE
    ]


def list_sidebar_projects(drive_root: Any) -> List[Dict[str, Any]]:
    """Projects visible while active or while deletion is quiescing."""
    return [
        project for project in list_reserved_projects(drive_root)
        if project.get("lifecycle") in {PROJECT_ACTIVE, PROJECT_DELETING}
    ]


# project_id -> canonical working_dir, memoized on the same registry version
# stamp as _CHAT_BINDING_INDEX (see that comment for what the stamp can and
# cannot prove).
_WORKING_DIR_INDEX: Dict[str, tuple] = {}


def project_working_dirs(drive_root: Any) -> Optional[Dict[str, str]]:
    """``project_id -> registered working_dir`` for every RESERVED project.

    The writer-lease lane resolver (``ouroboros.project_lease``) needs this and
    must stay filesystem-free under the supervisor queue lock, so the map is
    built HERE and handed in. A task scoped post-hoc through
    ``mark_task_project`` carries no ``workspace_root`` of its own; without this
    map its lane would not match the room task already writing the SAME folder
    and two top-level writers would enter it. File-less projects are omitted —
    the lease then keys such a task on its project alone, which is narrower than
    a folder lane and is documented as such rather than assumed equivalent.

    ``None`` means the folders are UNKNOWN, and it is a different fact from an
    empty map. ``{}`` says "no project has a registered folder", which the lane
    is entitled to act on: it keys those tasks on their project alone and
    serializes them against each other. An unreadable registry says nothing at
    all — and collapsing the two made a truncated write, a partial
    ``atomic_write`` on a full disk or a hand-edit read as "no project has a
    folder", after which a folder-bearing candidate was compared against a
    narrow lane, matched nothing, and a SECOND writer entered the folder (I3).
    No exception is required to reach that: the parse simply yields no
    ``projects`` list. So an existing registry path that does not parse into one,
    and any failure while reading it, answer ``None``; a registry that is not
    there yet answers ``{}``, because "this install has no projects" is a fact.

    Values are canonicalized here as well as at write time. New rows are stored
    resolved, but a registry written BEFORE that could hold an unresolved
    spelling — and against a task's already-realpath'd ``workspace_root`` that
    is a DIFFERENT string, i.e. a second concrete lane and the very split this
    map exists to close. The resolve is memoized on the registry file's version
    stamp, so the filesystem is touched once per registry write rather than once
    per assignment pass. ``None`` is deliberately NOT memoized: it describes a
    file this process could not read, and the next caller should look again.
    """
    path = _registry_path(drive_root)
    key = str(path)
    try:
        stat = path.stat()
        stamp: tuple = (stat.st_mtime_ns, stat.st_size, _REGISTRY_WRITE_SEQ.get(key, 0))
    except OSError:
        stamp = (0, 0, _REGISTRY_WRITE_SEQ.get(key, 0))
    cached = _WORKING_DIR_INDEX.get(key)
    if cached is not None and cached[0] == stamp:
        return dict(cached[1])
    out: Dict[str, str] = {}
    try:
        raw = read_json_dict(path)
        if path.exists() and not (isinstance(raw, dict) and isinstance(raw.get("projects"), list)):
            # The file is THERE and holds no project list. `_load` fails open to
            # `{"projects": []}` for every other reader, which is right for them
            # and wrong for the lane: see the docstring.
            log.warning("project_working_dirs: %s holds no readable projects list", path)
            return None
        for project in list_reserved_projects(drive_root):
            pid = str(project.get("id") or "")
            folder = _canonical_working_dir(project.get("working_dir"))
            if pid and folder:
                out[pid] = folder
    except Exception:
        log.debug("project_working_dirs failed", exc_info=True)
        return None
    _WORKING_DIR_INDEX[key] = (stamp, dict(out))
    return out


def reserved_project_chat_ids(drive_root: Any) -> set:
    """The set of chat_ids reserved by every Project lifecycle state.

    The TRUTH source for "is this chat a project thread" — a bare numeric range
    cannot disambiguate from large external-transport (e.g. Telegram) chat ids,
    so routing/history/UI classify by registry membership instead.

    NOT an isolation boundary (full project awareness, v6.32.0): the one identity
    sees ALL threads in its unified memory. This classifier drives (a) the UI
    history/fan-out partition that organizes threads into panels, (b) message
    routing, and (c) the project TASK's FOCUSED passive context (build_recent_
    sections shows the task its own thread).

    Covers EVERY thread of every project (thread #0 included, via the canonical
    projection) — one widening makes threads visible to history, ``/api/state``
    and the agent's context at once.
    """
    out = set()
    try:
        for project in list_reserved_projects(drive_root):
            try:
                out.update(_row_chat_ids(project))
            except (TypeError, ValueError):
                continue
    except Exception:
        log.debug("reserved_project_chat_ids failed", exc_info=True)
    out.discard(0)
    return out


def registered_project_chat_ids(drive_root: Any) -> set:
    """One-minor compatibility alias for :func:`reserved_project_chat_ids`."""
    key = str(pathlib.Path(drive_root).resolve(strict=False))
    if key not in _DEPRECATED_CHAT_IDS_EVENTS:
        _DEPRECATED_CHAT_IDS_EVENTS.add(key)
        try:
            from ouroboros.utils import append_jsonl

            append_jsonl(
                pathlib.Path(drive_root) / "logs" / "events.jsonl",
                {
                    "ts": utc_now_iso(),
                    "type": "deprecated_project_chat_ids_alias_used",
                    "alias": "registered_project_chat_ids",
                    "replacement": "reserved_project_chat_ids",
                },
            )
        except Exception:
            log.debug("Failed to record Project chat-id alias use", exc_info=True)
    return reserved_project_chat_ids(drive_root)


def get_project(drive_root: Any, project_id: str) -> Optional[Dict[str, Any]]:
    pid = sanitize_project_id(project_id)
    if not pid:
        return None
    for project in list_projects(drive_root):
        if project.get("id") == pid:
            return dict(project)
    return None


def get_reserved_project(drive_root: Any, project_id: str) -> Optional[Dict[str, Any]]:
    """Lookup irrespective of lifecycle (history/recovery only)."""
    pid = sanitize_project_id(project_id)
    if not pid:
        return None
    for project in list_reserved_projects(drive_root):
        if project.get("id") == pid:
            return dict(project)
    return None


# chat_id -> binding index, memoized on the registry file's
# (mtime_ns, size, in-process write counter). C4: "which project/thread owns this
# chat" is asked once per inbound message and per history request, and with
# threads the naive scan is projects x threads.
#
# Invalidation is a HEURISTIC, not a proof. atomic_write_json renames into place,
# so a write normally changes the mtime; but a same-size rewrite landing inside
# one filesystem timestamp tick (coarse mtime granularity on some filesystems)
# can produce an identical (mtime_ns, size) pair. The monotonic counter closes
# that window for writes made by THIS process — the case a request-path read is
# actually racing. A concurrent write by ANOTHER process within the same tick can
# still be missed for the remainder of it; the index is a routing cache, and the
# next differing stamp repairs it.
_CHAT_BINDING_INDEX: Dict[str, tuple] = {}
# Bumped by _save on every registry write; part of the memo stamp above.
_REGISTRY_WRITE_SEQ: Dict[str, int] = {}


def _chat_binding_index(drive_root: Any) -> Dict[int, Dict[str, Any]]:
    path = _registry_path(drive_root)
    key = str(path)
    try:
        stat = path.stat()
        stamp: tuple = (stat.st_mtime_ns, stat.st_size, _REGISTRY_WRITE_SEQ.get(key, 0))
    except OSError:
        stamp = (0, 0, _REGISTRY_WRITE_SEQ.get(key, 0))
    cached = _CHAT_BINDING_INDEX.get(key)
    if cached is not None and cached[0] == stamp:
        return cached[1]
    index: Dict[int, Dict[str, Any]] = {}
    for project in list_reserved_projects(drive_root):
        pid = str(project.get("id") or "")
        for thread in project_threads(project):
            index.setdefault(int(thread["chat_id"]), {
                "project_id": pid,
                "thread_id": int(thread["id"]),
                "chat_id": int(thread["chat_id"]),
                "lifecycle": str(project.get("lifecycle") or PROJECT_ACTIVE),
                # The THREAD's own lifecycle, beside the project's. Routing must
                # be able to refuse a fenced or tombstoned thread inside a
                # perfectly healthy project (X10) — reading only the project's
                # state would deliver messages into a room the owner deleted.
                "thread_lifecycle": str(thread.get("lifecycle") or THREAD_ACTIVE),
                "name": str(thread.get("name") or ""),
                "project": dict(project),
            })
    _CHAT_BINDING_INDEX[key] = (stamp, index)
    return index


def resolve_chat_binding(
    drive_root: Any, chat_id: Any, *, strict: bool = False
) -> Dict[str, Any]:
    """THE canonical "who owns this chat id" lookup (R3).

    Returns ``{project_id, thread_id, chat_id, lifecycle, thread_lifecycle,
    name, project}`` for ANY thread of ANY project in ANY lifecycle state, or
    ``{}`` for the main chat / an external transport id. Callers that must not
    resurrect a fenced room filter on BOTH lifecycles — a thread can be fenced or
    tombstoned inside a perfectly healthy project — and they must NOT compare a
    chat id against ``project["chat_id"]`` themselves; that comparison sees
    thread #0 only and misroutes every other thread to Main.

    ``strict=True`` makes an unreadable registry RAISE instead of answering ``{}``,
    and nothing else changes. Routing wants the fail-closed ``{}`` — a message that
    cannot be placed belongs in Main. But ``thread_history.thread_ancestry_lens``
    reads this to decide whether a chat HAS ancestors, and there ``{}`` for a read
    failure is a lie with consequences: an unreadable registry made a fork
    indistinguishable from Main, so its whole shared past vanished and the window
    still called itself complete (P6). One seam, two honest answers, rather than a
    second lookup that could drift from this one.
    """
    try:
        cid = int(chat_id or 0)
    except (TypeError, ValueError):
        return {}
    if not cid:
        return {}
    try:
        row = _chat_binding_index(drive_root).get(cid)
    except Exception:
        if strict:
            raise
        log.debug("resolve_chat_binding failed", exc_info=True)
        return {}
    return dict(row) if row else {}


def _canonical_working_dir(raw: Any) -> str:
    """Symlink-resolved spelling of a project's folder, stored at WRITE time.

    The writer-lease lane compares a task's ``workspace_root`` against this
    value and must stay filesystem-free under the queue lock (see
    ``ouroboros.project_lease``), so the one place allowed to touch the disk is
    here — the record write. ``workspace_admission.validate_workspace_root``
    already resolves the task-side carrier the same way, so both spellings meet
    as realpaths and the pure ``normpath``/``normcase`` comparison is enough.
    CASE is preserved: this value is also shown to the owner, and
    ``normcase`` would lowercase it on a case-insensitive filesystem.
    Fail-open — an unresolvable path is stored as given rather than dropped.
    """
    text = str(raw or "").strip()
    if not text:
        return ""
    try:
        return str(pathlib.Path(text).expanduser().resolve(strict=False))
    except (OSError, RuntimeError, ValueError):
        log.debug("working_dir canonicalization failed for %r", text, exc_info=True)
        return text


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
                if existing.get("lifecycle") != PROJECT_ACTIVE:
                    raise ValueError(
                        f"project id {pid!r} is permanently reserved by a "
                        f"{existing.get('lifecycle')} project"
                    )
                return dict(existing)
        # Registry-WIDE chat-id reservation (X1). A project's chat id is
        # deterministic from its id, so a collision cannot be retried away —
        # refuse loudly instead of silently merging two histories. Every
        # creation path funnels through here under the same file lock.
        chat_id = project_chat_id(pid)
        clash = _chat_id_owners(data["projects"]).get(chat_id)
        if clash:
            raise ValueError(
                f"chat id {chat_id} for project {pid!r} is already reserved by "
                f"{clash} — pick a different project id"
            )
        entry = {
            "id": pid,
            "name": _validated_name(name, pid),
            "chat_id": chat_id,
            "working_dir": _canonical_working_dir(working_dir),
            "origin": str(origin or "owner"),
            "created_at": utc_now_iso(),
            "last_active_at": utc_now_iso(),
            "lifecycle": PROJECT_ACTIVE,
            "routing_generation": 0,
            # Project-wide AGGREGATE (the flat project_seen_revision cursor)…
            "visible_revision": 0,
            # …and thread #0's OWN counter, which siblings must not advance.
            "thread0_visible_revision": 0,
            "delete_error": "",
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
    allowed = {
        "name", "working_dir", "last_active_at", "provenance", "clone_url", "trusted_at",
        # Write-once legacy-activity fact seeded by the boot-reconcile backfill
        # (_backfill_thread_activity); read by projects_summary's derivation.
        "thread_activity_seen",
    }
    with _file_write_lock(_registry_path(drive_root)):
        data = _load(drive_root)
        for entry in data["projects"]:
            if entry.get("id") != pid or entry.get("lifecycle") != PROJECT_ACTIVE:
                continue
            for key, value in updates.items():
                if key not in allowed:
                    continue
                if key == "name":
                    value = _validated_name(value, str(entry.get("id") or ""))
                elif key == "working_dir":
                    # Record-write-time canonicalization (see the helper): the
                    # lease compares this path purely, so symlinks resolve here.
                    value = _canonical_working_dir(value)
                entry[key] = value
            _save(drive_root, data)
            return dict(entry)
    return None


def set_working_dir_if_absent(
    drive_root: Any,
    project_id: str,
    working_dir: str,
    *,
    provenance: str = "",
    trusted_at: str = "",
) -> tuple[str, bool]:
    """Bind a project's working_dir ONLY if it has none — check and write under ONE lock.

    "Never overwrites an existing working_dir/provenance" was true of the code but
    not of the timeline: both places that promised it (``adopt_task_workspace``,
    ``ensure_project_workspace``) read the entry with ``get_project`` and wrote it
    back with ``update_project``, two separately-locked operations with an unbounded
    gap between them — a folder validation in one, a whole genesis provisioning in
    the other. Two writers interleaving there both saw "no working_dir" and both
    wrote, so the loser's folder was silently replaced; when the loser was
    ``ensure_project_workspace`` the abandoned genesis tree sits under the durable
    projects root, which is deliberately never GC-pruned, i.e. it is orphaned for
    good. Doing the test and the write inside the same ``_file_write_lock`` makes
    the promise atomic across threads AND processes.

    ``provenance``/``trusted_at`` follow the same historical-fact rule as elsewhere:
    written only when the row carries nothing (or ``"none"``) yet.

    Returns ``(effective_working_dir, claimed)`` — ``claimed`` is True only for the
    writer that actually bound the folder, so the loser can report the truth instead
    of assuming its own value landed. An unknown/inactive project answers ``("", False)``.
    """
    pid = sanitize_project_id(project_id)
    folder = str(working_dir or "").strip()
    if not pid or not folder:
        return "", False
    with _file_write_lock(_registry_path(drive_root)):
        data = _load(drive_root)
        for entry in data["projects"]:
            if entry.get("id") != pid or entry.get("lifecycle") != PROJECT_ACTIVE:
                continue
            existing = str(entry.get("working_dir") or "").strip()
            if existing:
                return existing, False
            entry["working_dir"] = folder
            if provenance and str(entry.get("provenance") or "").strip() in ("", "none"):
                entry["provenance"] = provenance
                if trusted_at and not str(entry.get("trusted_at") or "").strip():
                    entry["trusted_at"] = trusted_at
            _save(drive_root, data)
            return folder, True
    return "", False


def replace_working_dir_if_unchanged(
    drive_root: Any,
    project_id: str,
    expected: str,
    working_dir: str,
    *,
    provenance: str = "",
) -> tuple[str, bool]:
    """Compare-and-swap a project's working_dir — compare and write under ONE lock.

    The sibling of :func:`set_working_dir_if_absent` for the case that function
    deliberately declines: the row NAMES a folder which has since vanished, and
    replacing it is the whole point of the call. "Only if absent" is wrong there,
    but so was the unconditional ``update_project`` that replaced it, for the same
    reason ``set_working_dir_if_absent`` exists at all — the read and the write were
    separately locked with an entire genesis provisioning between them.

    Reproduced: two callers both observe the same vanished ``working_dir``, both
    provision (``genesis_1``, ``genesis_2``), both write, so the registry ends on
    one and the OTHER is orphaned under the never-pruned durable projects root —
    while BOTH callers are handed their own path back, so one of them reports a
    binding that does not exist. Second reproduction: an owner attach landing
    between the read and the write is silently overwritten.

    ``expected`` is the value the caller OBSERVED. The write happens only while the
    row still holds it; otherwise the row is left alone and the current value is
    returned. ``(effective_working_dir, claimed)`` — ``claimed`` is True only for
    the writer that actually swapped, so a loser reports the winner's path and logs
    its orphan instead of assuming its own value landed. An unknown or inactive
    project answers ``("", False)``, exactly as the sibling does.

    ``provenance`` follows the same historical-fact rule, and is judged from the row
    UNDER THE LOCK rather than from the caller's stale read: how a folder came to be
    is only written when the row carries nothing.
    """
    pid = sanitize_project_id(project_id)
    folder = str(working_dir or "").strip()
    want = str(expected or "").strip()
    if not pid or not folder:
        return "", False
    with _file_write_lock(_registry_path(drive_root)):
        data = _load(drive_root)
        for entry in data["projects"]:
            if entry.get("id") != pid or entry.get("lifecycle") != PROJECT_ACTIVE:
                continue
            existing = str(entry.get("working_dir") or "").strip()
            if existing != want:
                # Somebody else moved it: an owner attach, or another provisioner
                # that won. Either way this caller's observation is stale and its
                # write would destroy a NEWER binding.
                return existing, False
            entry["working_dir"] = folder
            if provenance and str(entry.get("provenance") or "").strip() in ("", "none"):
                entry["provenance"] = provenance
            _save(drive_root, data)
            return folder, True
    return "", False


def begin_project_deletion(drive_root: Any, project_id: str) -> Optional[Dict[str, Any]]:
    """Close admission/routing before the supervisor cancels the live subtree."""
    pid = sanitize_project_id(project_id)
    if not pid:
        return None
    with _file_write_lock(_registry_path(drive_root)):
        data = _load(drive_root)
        for entry in data["projects"]:
            if entry.get("id") != pid:
                continue
            if entry.get("lifecycle") in {PROJECT_DELETING, PROJECT_TOMBSTONED}:
                return dict(entry)
            entry["lifecycle"] = PROJECT_DELETING
            entry["routing_generation"] = int(entry.get("routing_generation") or 0) + 1
            entry["admission_closed_at"] = utc_now_iso()
            entry["deleting_at"] = entry["admission_closed_at"]
            entry["delete_error"] = ""
            _save(drive_root, data)
            return dict(entry)
    return None


def fail_project_deletion(
    drive_root: Any, project_id: str, error: str
) -> Optional[Dict[str, Any]]:
    """Keep a fenced Project recoverably deleting while quiescence is pending."""
    pid = sanitize_project_id(project_id)
    if not pid:
        return None
    with _file_write_lock(_registry_path(drive_root)):
        data = _load(drive_root)
        for entry in data["projects"]:
            if entry.get("id") == pid and entry.get("lifecycle") == PROJECT_DELETING:
                entry["delete_error"] = truncate_review_artifact(
                    str(error or "deletion did not quiesce"), limit=DELETE_ERROR_LIMIT,
                )
                _save(drive_root, data)
                return dict(entry)
    return None


def complete_project_deletion(
    drive_root: Any, project_id: str, *, delete_error: str = ""
) -> Optional[Dict[str, Any]]:
    """Commit the tombstone after the supervisor proves subtree quiescence.

    ``delete_error`` records what the teardown could NOT take — today, thread
    checkouts that outlived the sweep. The tombstone still happens: keeping the
    project alive because a folder survived was considered and REJECTED by the
    owner (it collides with §I M2 and with "deleting a thread with its worktree
    must be easy"). But a tombstoned project is invisible on every surface, so a
    surviving checkout would be a folder and a ``thread/*`` branch nothing can
    reach, previously announced by a ``log.warning`` alone. It is written onto the
    row here, and the caller also tells the owner in chat — never silently.

    The field is BOUNDED, never silently cut: a flat ``[:2000]`` over a note that
    names surviving folders and branches dropped whole checkouts out of the only
    record that points at them, and ended the sentence mid-word. The caller
    already bounds its list at an entry boundary and declares what it left out;
    this is the string backstop for anything else written here, and it goes
    through the ``truncate_review_artifact`` SSOT so an overflow arrives with its
    omission marker attached (DEVELOPMENT.md "No silent truncation").
    """
    pid = sanitize_project_id(project_id)
    if not pid:
        return None
    note = truncate_review_artifact(str(delete_error or ""), limit=DELETE_ERROR_LIMIT)
    with _file_write_lock(_registry_path(drive_root)):
        data = _load(drive_root)
        for entry in data["projects"]:
            if entry.get("id") != pid:
                continue
            if entry.get("lifecycle") == PROJECT_TOMBSTONED:
                return dict(entry)
            if entry.get("lifecycle") != PROJECT_DELETING:
                raise ValueError(f"project {pid!r} is not deleting")
            entry["lifecycle"] = PROJECT_TOMBSTONED
            entry["tombstoned_at"] = utc_now_iso()
            entry["delete_error"] = note
            _save(drive_root, data)
            log.info(
                "Project tombstoned: %s (history, bindings, folder and memory preserved)%s",
                pid, f" — LEFT BEHIND: {note}" if note else "",
            )
            return dict(entry)
    return None


def delete_project(drive_root: Any, project_id: str) -> bool:
    """Compatibility completion; live deletion must first erect its queue fence."""
    row = get_reserved_project(drive_root, project_id)
    if row is None:
        return False
    if row.get("lifecycle") == PROJECT_TOMBSTONED:
        return True
    if row.get("lifecycle") != PROJECT_DELETING:
        raise RuntimeError("live Project deletion requires cancellation/quiescence first")
    complete_project_deletion(drive_root, project_id)
    return True


def increment_project_visible_revision(
    drive_root: Any,
    *,
    project_id: str = "",
    chat_id: Any = 0,
) -> Optional[Dict[str, Any]]:
    """Advance unread state for one newly-appended owner-visible canonical row.

    EVERY thread — thread #0 included — advances its OWN counter AND the
    project's aggregate. The aggregate is what today's flat
    ``project_seen_revision`` cursor compares against, so leaving it untouched
    would make a non-primary thread's activity silently unread-invisible; the
    per-thread counters are the numbers T1's nested cursor will read. Thread #0
    keeps its own number in ``thread0_visible_revision`` rather than borrowing
    the aggregate — otherwise a message in ANY sibling thread would also mark
    thread #0 unread.
    """
    pid = sanitize_project_id(project_id)
    try:
        cid = int(chat_id or 0)
    except (TypeError, ValueError):
        cid = 0
    if not pid and not cid:
        return None
    with _file_write_lock(_registry_path(drive_root)):
        data = _load(drive_root)
        for entry in data["projects"]:
            if entry.get("lifecycle") != PROJECT_ACTIVE:
                continue
            thread_hit = None
            if cid:
                thread_hit = next(
                    (t for t in project_threads(entry) if int(t["chat_id"]) == cid), None
                )
            if not ((pid and entry.get("id") == pid) or thread_hit is not None):
                continue
            entry["visible_revision"] = int(entry.get("visible_revision") or 0) + 1
            hit_id = int(thread_hit["id"]) if thread_hit is not None else MAIN_THREAD_ID
            if hit_id == MAIN_THREAD_ID:
                # Thread #0's own counter. `entry` came through
                # _normalize_project_row, which seeds a legacy row's value from
                # the aggregate, so this never silently restarts at 0.
                entry["thread0_visible_revision"] = int(
                    entry.get("thread0_visible_revision") or 0
                ) + 1
            else:
                threads = _normalize_thread_rows(entry.get("threads"))
                for row in threads:
                    if int(row["id"]) == hit_id:
                        row["visible_revision"] = int(row.get("visible_revision") or 0) + 1
                entry["threads"] = threads
            _save(drive_root, data)
            return dict(entry)
    return None


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
        if projects_root.is_dir():
            with _file_write_lock(_registry_path(drive_root)):
                data = _load(drive_root)
                known = {p.get("id") for p in data["projects"]}
                reserved = _chat_id_owners(data["projects"])
                for entry in sorted(projects_root.iterdir()):
                    if not entry.is_dir() or entry.name.startswith("."):
                        continue
                    pid = sanitize_project_id(entry.name)
                    if not pid or pid in known:
                        continue
                    # Same registry-wide reservation invariant as create_project
                    # (X1): reconcile mints hashed ids too, so an unchecked
                    # append here could collide with an existing project OR
                    # thread. Skip loudly — a reconcile must never merge two
                    # histories, and the store stays on disk for the owner.
                    chat_id = project_chat_id(pid)
                    if chat_id in reserved:
                        log.error(
                            "Project reconcile SKIPPED %s: chat id %s already reserved by %s",
                            pid, chat_id, reserved[chat_id],
                        )
                        continue
                    reserved[chat_id] = [(pid, MAIN_THREAD_ID)]
                    data["projects"].append({
                        "id": pid,
                        "name": pid,
                        "chat_id": chat_id,
                        "working_dir": "",
                        "origin": "reconcile",
                        "created_at": utc_now_iso(),
                        "last_active_at": utc_now_iso(),
                        "lifecycle": PROJECT_ACTIVE,
                        "routing_generation": 0,
                        "visible_revision": 0,
                        "delete_error": "",
                    })
                    known.add(pid)
                    added += 1
                if added:
                    _save(drive_root, data)
                    log.info("Project registry reconcile: %d store(s) registered", added)
    except Exception:
        log.warning("Project registry reconcile failed", exc_info=True)
    _backfill_thread_activity(drive_root)
    return added


# Drive roots whose legacy thread-activity backfill already ran in this process.
# The scan is a boot-reconcile concern: once per process per root is enough,
# because everything AFTER the backfill is covered by the registry-facts
# derivation in projects_summary (visible_revision/bindings/origin).
# Named "thread activity", but it stays in the REGISTRY on purpose: it touches no
# thread row at all — it reads the project's own chat_id (thread #0), is driven by
# reconcile_projects, and writes through update_project. Moving it to the thread
# module would borrow four registry internals across a seam it never crosses.
_ACTIVITY_BACKFILL_DONE: set = set()


def _backfill_thread_activity(drive_root: Any) -> int:
    """One-time archive-aware seeding of the durable ``thread_activity_seen`` flag.

    ``projects_summary`` derives thread activity from registry facts alone
    (origin, bindings, ``visible_revision``) — but a legacy project whose
    activity predates the ``visible_revision`` counter would read inactive
    forever. So the boot reconcile scans live + archived chat/progress logs
    ONCE per process for a row carrying each such project's chat_id, and
    persists a write-once flag through the registry's own write path
    (``update_project``). This never runs on the GET path, never removes the
    flag, and a scan that finds nothing simply leaves the project inactive.
    The done-marker is set only AFTER a successful scan/write pass, so a
    transiently failed backfill retries on the next reconcile tick instead of
    silently waiting for a process restart.
    """
    key = str(pathlib.Path(drive_root).resolve(strict=False))
    if key in _ACTIVITY_BACKFILL_DONE:
        return 0
    flagged = 0
    try:
        bindings = _load_bindings(drive_root).get("bindings", {})
        bound_pids = {
            str(row.get("project_id") or "")
            for row in bindings.values()
            if isinstance(row, dict)
        }
        candidates: Dict[int, str] = {}
        with _LOCK:
            projects = _load(drive_root)["projects"]
        for project in projects:
            # update_project persists only ACTIVE rows; deleting/tombstoned
            # projects keep deriving from origin/bindings/visible_revision.
            if project.get("lifecycle") != PROJECT_ACTIVE:
                continue
            if project.get("thread_activity_seen"):
                continue
            pid = str(project.get("id") or "")
            if (
                not pid
                or str(project.get("origin") or "") == "owner_ui"
                or int(project.get("visible_revision") or 0) > 0
                or pid in bound_pids
            ):
                continue  # already active by derivation — no flag needed
            try:
                cid = int(project.get("chat_id") or 0)
            except (TypeError, ValueError):
                cid = 0
            if cid:
                candidates[cid] = pid
        if not candidates:
            _ACTIVITY_BACKFILL_DONE.add(key)
            return 0
        logs_dir = pathlib.Path(drive_root) / "logs"
        archive_dir = pathlib.Path(drive_root) / "archive"
        paths = [logs_dir / "chat.jsonl", logs_dir / "progress.jsonl"]
        if archive_dir.is_dir():
            paths.extend(sorted(archive_dir.glob("chat_*.jsonl"), reverse=True))
            paths.extend(sorted(archive_dir.glob("progress_*.jsonl"), reverse=True))
        seen: set = set()
        for path in paths:
            if len(seen) == len(candidates):
                break
            if not path.is_file():
                continue
            try:
                for row in iter_jsonl_objects(path):
                    try:
                        cid = int(row.get("chat_id") or 1)
                    except (TypeError, ValueError):
                        continue
                    if cid in candidates and cid not in seen:
                        seen.add(cid)
                        if len(seen) == len(candidates):
                            break
            except Exception:
                log.debug("thread-activity backfill scan failed for %s", path, exc_info=True)
        for cid in sorted(seen):
            if update_project(drive_root, candidates[cid], thread_activity_seen=True) is not None:
                flagged += 1
        if flagged:
            log.info("Thread-activity backfill: %d legacy project(s) flagged", flagged)
        _ACTIVITY_BACKFILL_DONE.add(key)
    except Exception:
        log.warning("Thread-activity backfill failed", exc_info=True)
    return flagged


def ensure_project_workspace(drive_root: Any, project_id: str, repo_dir: Any) -> str:
    """Provision (once) an invisible-git working folder for a project.

    Reuses the genesis-project machinery: a standalone git repo under the
    durable projects root (never GC-pruned, isolated from repo/ and data/).
    Returns the absolute path ("" when provisioning failed). File-less
    projects simply never call this.

    The folder is stamped ``provenance="genesis"`` in the SAME write that binds
    it (A11). Without that stamp the row said only that the project has SOME
    working_dir, which is what made the auto-provisioned place invisible: the
    owner asked for a project, a folder appeared somewhere under the durable
    projects root, and no surface could tell them Ouroboros had made it rather
    than that they had pointed at it themselves. An existing provenance is never
    overwritten — how a folder came to be is a historical fact, and this branch
    only runs when the project had no usable folder at all.

    Provisioning is slow (a real ``git init`` + seed commit), so a FIRST bind is
    ATOMIC (``set_working_dir_if_absent``, T2-7) rather than a second unlocked
    write: if another writer bound a place while this one was digging, the winner's
    folder is returned untouched and the abandoned genesis tree is LOGGED — it lives
    under the durable projects root, which is never GC-pruned, so an unreported loss
    would be an orphan forever.

    A REPLACEMENT is a different write, but it is not an UNCONDITIONAL one. This
    function also runs when the row already names a folder that has since VANISHED,
    and there the "only if absent" rule is exactly wrong: it declines, the caller
    reports a concurrent bind that never happened, and the path handed back is the
    non-existent one — while the tree just provisioned is orphaned under a root
    nothing ever prunes. So the empty case claims atomically through
    ``set_working_dir_if_absent`` and the stale case COMPARE-AND-SWAPS through
    ``replace_working_dir_if_unchanged``, writing only while the row still holds the
    path that was observed vanished. A plain ``update_project`` there had the same
    read-then-write gap the atomic bind exists to close: two callers both observed
    the same stale path, both provisioned, both wrote, one durable tree was orphaned
    for good and both callers were told their own path had been bound — and an owner
    attach landing in that window was silently overwritten. Provenance still follows
    the historical-fact rule and is now judged under the lock: it is stamped only
    when the row carries nothing.
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
        if existing:
            # The row names a folder; it is simply GONE. Replacing it is the whole
            # point of this call — but as a COMPARE-AND-SWAP against the value that
            # was observed vanished, not as an unconditional write. Two callers both
            # observing the same stale path both provisioned and both wrote, so one
            # durable tree was orphaned under the never-pruned projects root while
            # BOTH were told their own path had been bound; and an owner attach
            # landing in that window was silently overwritten.
            bound, claimed = replace_working_dir_if_unchanged(
                drive_root, entry["id"], existing, str(handle.path),
                provenance="genesis",
            )
            if claimed:
                return str(handle.path)
            if bound:
                log.warning(
                    "Project %s was re-bound concurrently (%s); the genesis tree "
                    "provisioned here is abandoned and NOT auto-removed: %s",
                    entry["id"], bound, handle.path,
                )
                return bound
            log.warning(
                "Project %s is no longer active; genesis tree %s was not bound",
                entry["id"], handle.path,
            )
            return ""
        bound, claimed = set_working_dir_if_absent(
            drive_root, entry["id"], str(handle.path), provenance="genesis"
        )
        if claimed:
            return str(handle.path)
        if bound:
            log.warning(
                "Project %s was given a folder concurrently (%s); the genesis tree "
                "provisioned here is abandoned and NOT auto-removed: %s",
                entry["id"], bound, handle.path,
            )
            return bound
        # A project that stopped being active mid-provision: the entry that asked
        # for the folder is gone, so there is nothing to bind it to.
        log.warning(
            "Project %s is no longer active; genesis tree %s was not bound",
            entry["id"], handle.path,
        )
        return ""
    except Exception:
        log.warning("Project workspace provisioning failed for %s", project_id, exc_info=True)
        return ""


def adopt_task_workspace(
    drive_root: Any, project_id: str, workspace_root: Any, *, system_repo_dir: Any
) -> tuple[str, str]:
    """Give a project born FROM A TASK the folder that task was already working in.

    A11: a project must have a designated place. Both conversion paths — the UI
    "turn into project" card and the in-task ``ensure_project_scope`` — used to
    register a project and drop the task's ``workspace_root`` on the floor, so a
    project made out of work happening in a real folder came out folder-less and
    the NEXT task in it auto-provisioned a different, empty tree somewhere else.
    The task's own folder is the obvious place; nothing else has a better claim.

    Adopting is an ATTACH, so it re-runs the attach guards rather than trusting the
    task record: the same resolved-realpath checks (exists, real dir, not the home
    root, disjoint from the Ouroboros repo/data roots, not nested inside another
    repository), and deliberately NO git requirement (A12 — a plain folder is a
    legitimate place). An EXISTING working_dir is never overwritten; a project that
    already has a place keeps it.

    It also applies a rule the attach surfaces do not need. An attach path is typed
    by the owner; an ADOPTED path arrives from a task record, and a task's workspace
    is precisely where Ouroboros's own EPHEMERAL checkouts live — an acting
    subagent's ``self_worktree``, a thread's branch-off worktree. Those are linked
    worktrees and age-swept roots: adopting one would give the project a place that
    a ``git worktree remove`` or a retention sweep can delete underneath it. So
    ``ephemeral_checkout_reason`` refuses them through the same disclosure channel.

    Returns ``(working_dir, error)``. ``error`` is a disclosure, not a failure the
    caller must abort on: conversion's job is to create the project, and a folder
    that has since moved is worth reporting rather than either hiding or fatal.
    """
    pid = sanitize_project_id(project_id)
    raw = str(workspace_root or "").strip()
    if not pid or not raw:
        return "", ""
    entry = get_project(drive_root, pid)
    if entry is None:
        return "", ""
    if str(entry.get("working_dir") or "").strip():
        return str(entry["working_dir"]), ""
    from ouroboros.project_sources import ephemeral_checkout_reason, validate_attach_path

    resolved, error = validate_attach_path(
        raw, system_repo_dir=system_repo_dir, drive_root=drive_root
    )
    if error or resolved is None:
        return "", f"the task's workspace {raw} was not adopted as the project folder: {error}"
    ephemeral = ephemeral_checkout_reason(resolved)
    if ephemeral:
        return "", f"the task's workspace {raw} was not adopted as the project folder: {ephemeral}"
    # The owner chose this folder when they started the work there; the conversion
    # inherits that grant rather than asking for it a second time. The claim is
    # ATOMIC (T2-7): the "does it already have a place" check and the write happen
    # inside ONE registry lock, so a conversion racing an auto-provision cannot
    # overwrite the winner's folder and orphan a genesis tree nothing ever GCs.
    bound, claimed = set_working_dir_if_absent(
        drive_root, pid, str(resolved), provenance="attached", trusted_at=utc_now_iso()
    )
    if claimed:
        return str(resolved), ""
    if bound:
        # Another writer bound a place first. That project HAS its place; saying so
        # is the truth, and overwriting it would be the very race this closes.
        return bound, ""
    return "", (
        f"the task's workspace {raw} was not adopted as the project folder: "
        f"project {pid!r} is no longer registered as active"
    )


def projects_summary(
    drive_root: Any, *, limit: int = 50, live_chat_ids: Any = None,
    include_archived: bool = False,
) -> List[Dict[str, Any]]:
    """Compact list for /api/state and the sidebar.

    ``live_chat_ids`` is the set of chat ids with a task running right now, and
    the ONLY reason this projection takes it: an ARCHIVED thread stays VISIBLE
    until its task is terminal (X10), because hiding a room that is still
    emitting output leaves the owner watching nothing while work continues. The
    caller supplies it because reading the supervisor queue belongs at the
    gateway, not inside a registry projection that every read path touches;
    omitting it simply means no archived thread is treated as live.

    ``include_archived`` asks for them ON PURPOSE, and it exists because this is
    the ONLY projection that lists threads. With archived ones filtered out of it
    unconditionally, no surface the owner could reach ever carried an archived
    thread, which made ``POST …/restore`` and the ``restore`` row in the thread
    menu unreachable BY CONSTRUCTION: archive was a one-way trip. Restoring
    something requires a surface that can show it, so the caller asks for one.
    """
    out: List[Dict[str, Any]] = []
    bindings = _load_bindings(drive_root).get("bindings", {})

    def _has_thread_activity(project: Dict[str, Any]) -> bool:
        # Registry-facts derivation ONLY — the GET path never scans logs and
        # never writes. Micro-delta vs the retired per-request log scan
        # (disclosed): a project whose thread carries ONLY telemetry rows (no
        # owner-visible canonical row, no binding) reads inactive until its
        # first visible row — exactly the junk-row shape this filter exists
        # to hide. Legacy projects whose activity predates the
        # visible_revision counter are covered by the write-once
        # `thread_activity_seen` flag seeded at boot reconcile
        # (_backfill_thread_activity).
        pid = str(project.get("id") or "")
        # v6.59.0: a project the OWNER explicitly created in the UI is always shown —
        # the activity filter exists to hide junk reconcile rows, not a fresh project
        # the owner just made (which has no chat rows yet by definition).
        if str(project.get("origin") or "") == "owner_ui":
            return True
        if any(isinstance(row, dict) and row.get("project_id") == pid for row in bindings.values()):
            return True
        if int(project.get("visible_revision") or 0) > 0:
            return True
        return bool(project.get("thread_activity_seen"))

    for project in list_sidebar_projects(drive_root)[: max(1, int(limit))]:
        out.append({
            "id": project.get("id"),
            "name": project.get("name"),
            "chat_id": project.get("chat_id"),
            "working_dir": project.get("working_dir") or "",
            "provenance": project.get("provenance") or "",
            "last_active_at": project.get("last_active_at") or "",
            "lifecycle": project.get("lifecycle") or PROJECT_ACTIVE,
            "routing_generation": int(project.get("routing_generation") or 0),
            "visible_revision": int(project.get("visible_revision") or 0),
            "delete_error": project.get("delete_error") or "",
            "has_thread_activity": _has_thread_activity(project),
            # Canonical projection, thread #0 first (X7). ``chat_id`` above stays
            # its compatibility alias, so a client that never learns about
            # threads keeps working unchanged. Archived and tombstoned threads
            # are FILTERED here rather than at every consumer — a surface that
            # forgot the filter would show the owner a thread they archived, and
            # one that hard-coded it would hide a live archived thread (X10).
            "threads": [
                thread for thread in project_threads(project)
                if thread_is_visible(thread, live_chat_ids, include_archived=include_archived)
            ],
        })
    return out


__all__ = [
    "PROJECT_ACTIVE",
    "PROJECT_DELETING",
    "PROJECT_NAME_MAX",
    "PROJECT_TOMBSTONED",
    "THREAD_NAME_MAX",
    "adopt_task_workspace",
    "all_task_bindings",
    "begin_project_deletion",
    "bind_task_to_project",
    "complete_project_deletion",
    "create_project",
    "THREAD_ACTIVE",
    "THREAD_ARCHIVED",
    "THREAD_DELETING",
    "THREAD_TOMBSTONED",
    "ThreadLifecycleError",
    "archive_thread",
    "begin_thread_deletion",
    "complete_thread_deletion",
    "create_thread",
    "fail_thread_deletion",
    "delete_project",
    "duplicate_chat_ids",
    "ensure_project_workspace",
    "fail_project_deletion",
    "fork_thread",
    "get_project",
    "get_reserved_project",
    "get_thread",
    "project_threads",
    "rename_thread",
    "restore_thread",
    "thread_is_visible",
    "resolve_chat_binding",
    "increment_project_visible_revision",
    "list_projects",
    "list_reserved_projects",
    "list_sidebar_projects",
    "project_binding_for_task",
    "project_chat_for_task",
    "project_thread_note_for_task",
    "project_chat_for_task_tree",
    "project_task_bindings",
    "project_working_dirs",
    "registered_project_chat_ids",
    "replace_working_dir_if_unchanged",
    "reserved_project_chat_ids",
    "projects_summary",
    "reconcile_projects",
    "set_working_dir_if_absent",
    "touch_project",
    "update_project",
]

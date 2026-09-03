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

from ouroboros.contracts.chat_id_policy import project_chat_id
from ouroboros.contracts.schema_versions import with_schema_version
from ouroboros.project_facts import sanitize_project_id
from ouroboros.utils import atomic_write_json, iter_jsonl_objects, read_json_dict, utc_now_iso, strip_markdown

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
PROJECT_LIFECYCLES = frozenset({PROJECT_ACTIVE, PROJECT_DELETING, PROJECT_TOMBSTONED})
_DEPRECATED_CHAT_IDS_EVENTS: set[str] = set()


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


_BINDINGS_LENS_CACHE: Dict[str, tuple] = {}


def _bindings_lens(drive_root: Any) -> Dict[str, Any]:
    """mtime/size-cached read of the bindings store for per-frame routing.

    The live-log addressing seam resolves lineage on every forwarded event
    (DEVELOPMENT "projection over replay": a per-interaction reader must not
    reparse a growing store), so the parse is keyed to the file identity the
    same way ``project_thread_chat_ids`` caches the registry."""
    path = _bindings_path(drive_root)
    try:
        st = path.stat()
        stamp: Any = (st.st_mtime_ns, st.st_size)
    except OSError:
        stamp = None
    key = str(path)
    cached = _BINDINGS_LENS_CACHE.get(key)
    if cached is not None and cached[0] == stamp:
        return cached[1]
    bindings = _load_bindings(drive_root)["bindings"]
    _BINDINGS_LENS_CACHE[key] = (stamp, bindings)
    return bindings


def project_chat_for_task_tree(
    drive_root: Any, task_id: Any, parent_task_id: Any = "", root_task_id: Any = ""
) -> int:
    """Resolve the project chat for a task by its TASK TREE: the task's OWN binding
    wins; else inherit from its parent; else its root. A subagent is never bound
    itself, so this is how its live frames + history are recognized as belonging to
    its root's project and route to the project thread instead of staying in the main
    chat (the cyber-racing "subagents vanished from the project" gap). Membership is
    DERIVED from lineage — no per-child binding store, one SSOT."""
    # One cached bindings view for all three probes (runs per live log event).
    try:
        bindings = _bindings_lens(drive_root)
    except Exception:
        log.debug("project_chat_for_task_tree bindings read failed", exc_info=True)
        return 0
    for tid in (task_id, parent_task_id, root_task_id):
        tid = str(tid or "").strip()
        if not tid:
            continue
        row = bindings.get(tid)
        if not isinstance(row, dict):
            continue
        try:
            chat = int(row.get("project_chat_id") or 0)
        except (TypeError, ValueError):
            chat = 0
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
    """
    out = set()
    try:
        for project in list_reserved_projects(drive_root):
            try:
                out.add(int(project.get("chat_id") or 0))
            except (TypeError, ValueError):
                continue
    except Exception:
        log.debug("reserved_project_chat_ids failed", exc_info=True)
    out.discard(0)
    return out


_THREAD_LENS_CACHE: Dict[str, tuple] = {}


def project_thread_chat_ids(drive_root: Any) -> frozenset:
    """Per-frame lens over :func:`reserved_project_chat_ids` for the live bus.

    The broadcast choke (``supervisor.message_bus``) stamps every outbound
    frame whose final ``chat_id`` is a reserved Project thread, so the browser
    can keep a project it has not learned yet out of Main. One ``stat`` per
    frame; the set is rebuilt only when the registry file changes.
    """
    path = _registry_path(drive_root)
    try:
        st = path.stat()
        stamp: Any = (st.st_mtime_ns, st.st_size)
    except OSError:
        stamp = None
    key = str(path)
    cached = _THREAD_LENS_CACHE.get(key)
    if cached is not None and cached[0] == stamp:
        return cached[1]
    ids = frozenset(reserved_project_chat_ids(drive_root))
    _THREAD_LENS_CACHE[key] = (stamp, ids)
    return ids


def stamp_project_thread(drive_root: Any, payload: Dict[str, Any]) -> None:
    """Set ``project_thread=True`` iff the payload's FINAL chat_id is a reserved
    Project thread. Called by the live broadcast choke (``supervisor.message_bus``)
    AFTER the envelope literal is built (constant-key assignment — the WS
    envelope-parity scanner forbids ``**`` widening inside the literal), so
    Main's fan-out can reject a not-yet-learned Project frame. Registry
    MEMBERSHIP, never a numeric range: external transport ids (Telegram) are
    not members and stay unstamped.
    """
    try:
        chat_id = int(payload.get("chat_id") or 0)
        if drive_root is not None and chat_id in project_thread_chat_ids(drive_root):
            payload["project_thread"] = True
    except Exception:
        log.debug("Project thread lens failed", exc_info=True)


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


def _bounded_presentation_name(value: Any, *, fallback: str = "") -> str:
    name = " ".join(str(value or "").split()) or str(fallback or "")
    return name if len(name) <= PROJECT_NAME_MAX else name[: PROJECT_NAME_MAX - 1].rstrip() + "…"


def task_presentation_snapshot(drive_root: Any, task_id: str, *, task: Any = None,
                               result: Any = None, project_id: str = "") -> Dict[str, Any]:
    tid = str(task_id or "").strip()
    sources = [row for row in (task, result) if isinstance(row, dict)]
    if tid:
        try:
            from ouroboros.task_status import load_effective_task_result
            stored = load_effective_task_result(
                pathlib.Path(drive_root), tid, materialize_artifacts=False)
            if isinstance(stored, dict):
                sources.append(stored)
        except Exception:
            log.debug("task presentation result lookup failed for %s", tid, exc_info=True)
        snapshot = read_json_dict(pathlib.Path(drive_root) / "state" / "queue_snapshot.json") or {}
        for row in [*list(snapshot.get("running") or []), *list(snapshot.get("pending") or [])]:
            queued = row.get("task") if isinstance(row, dict) else None
            if isinstance(queued, dict) and str(queued.get("id") or row.get("id") or "") == tid:
                sources.append(queued)
                break
    pid = str(project_id or "").strip()
    if not pid:
        for source in sources:
            pid = str(source.get("project_id") or "").strip()
            if pid:
                break
    if not pid and tid:
        binding = project_binding_for_task(drive_root, tid) or {}
        pid = str(binding.get("project_id") or "").strip()
    pname = ""
    registered = False
    if pid:
        # ONE registry read serves both the display name and the additive
        # ``project_routable`` fact: a workspace-derived proj_<hash> is
        # project-SCOPED without having a room, and a producer that announces it
        # would point the owner at a project that does not exist.
        project = get_reserved_project(drive_root, pid) or {}
        # ROUTABLE, not merely reserved: list_reserved_projects deliberately
        # includes deleting/tombstoned history reservations, and those have no
        # room left to open.
        registered = str(project.get("lifecycle") or "") == PROJECT_ACTIVE
        pname = _bounded_presentation_name(project.get("name"))
    if pname == pid:
        pname = ""
    if pid and not pname:
        pname = "Project"
    task_name = ""
    for field in ("title", "suggested_name", "objective", "description"):
        for source in sources:
            # Strip markdown BEFORE the name is flattened: the task half is a raw
            # request line, and ARCHITECTURE promises this label is plain text.
            task_name = _bounded_presentation_name(strip_markdown(str(source.get(field) or "")))
            if task_name:
                break
        if task_name:
            break
    task_name = task_name or "Task"
    return {"project_id": pid, "project_name": pname, "task_id": tid,
            "project_routable": registered, "task_name": task_name,
            "target_label": f"{pname} › {task_name}" if pname else task_name}


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

    The returned dict carries an additive ``created`` key — ``True`` only when
    THIS call registered the row, ``False`` on the idempotent replay of an
    existing project. Callers that need "did a project actually come into
    existence" (e.g. the agent-initiated ``project_started`` announcement)
    branch on it; the key is never persisted into the registry file.
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
                return {**existing, "created": False}
        entry = {
            "id": pid,
            "name": _validated_name(name, pid),
            "chat_id": project_chat_id(pid),
            "working_dir": str(working_dir or "").strip(),
            "origin": str(origin or "owner"),
            "created_at": utc_now_iso(),
            "last_active_at": utc_now_iso(),
            "lifecycle": PROJECT_ACTIVE,
            "routing_generation": 0,
            "visible_revision": 0,
            "delete_error": "",
        }
        data["projects"].append(entry)
        _save(drive_root, data)
        log.info("Project registered: %s (chat_id=%s)", pid, entry["chat_id"])
        return {**entry, "created": True}


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
        # Durable pointer to the project's newest finalized task result
        # (stamped by record_task_finalization; read first by
        # server._latest_project_task_result before its bounded scan).
        "last_task_result_id",
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
                entry[key] = value
            _save(drive_root, data)
            return dict(entry)
    return None


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
                entry["delete_error"] = str(error or "deletion did not quiesce")[:2000]
                _save(drive_root, data)
                return dict(entry)
    return None


def complete_project_deletion(drive_root: Any, project_id: str) -> Optional[Dict[str, Any]]:
    """Commit the tombstone after the supervisor proves subtree quiescence."""
    pid = sanitize_project_id(project_id)
    if not pid:
        return None
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
            entry["delete_error"] = ""
            _save(drive_root, data)
            try:
                # The store's `.project.json` is REGISTRY-AUTHORED recovery metadata,
                # not the owner's memory: leaving it would let a later loss of
                # projects.json (rows AND tombstones live in that one file) resurrect
                # the deleted room as active through marker recovery.
                (pathlib.Path(drive_root) / "projects" / pid / _PROJECT_MARKER_NAME).unlink(
                    missing_ok=True)
            except OSError:
                log.warning("could not remove %s marker of tombstoned %s", _PROJECT_MARKER_NAME, pid)
            log.info(
                "Project tombstoned: %s (history, bindings, folder and memory preserved)",
                pid,
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
    """Advance unread state for one newly-appended owner-visible canonical row."""
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
            try:
                matches_chat = cid and int(entry.get("chat_id") or 0) == cid
            except (TypeError, ValueError):
                matches_chat = False
            if (pid and entry.get("id") == pid) or matches_chat:
                entry["visible_revision"] = int(entry.get("visible_revision") or 0) + 1
                _save(drive_root, data)
                return dict(entry)
    return None


def touch_project(drive_root: Any, project_id: str) -> None:
    """Record activity (never raises)."""
    try:
        update_project(drive_root, project_id, last_active_at=utc_now_iso())
    except Exception:
        log.debug("touch_project failed for %s", project_id, exc_info=True)


_PROJECT_MARKER_NAME = ".project.json"


def _read_store_marker(store_dir: pathlib.Path) -> Dict[str, Any]:
    """Registry-authored provenance echo inside a project store ({} when absent or
    malformed). Never authoritative over a live registry row — it exists so reconcile
    can tell a REAL room's store from a workspace-task facts store in the one scenario
    the registry row itself is what was lost (the two are byte-shape identical on disk,
    and a non-ASCII display name hashes to the same ``proj_<hash>`` id namespace)."""
    marker = read_json_dict(store_dir / _PROJECT_MARKER_NAME)
    if not isinstance(marker, dict) or not isinstance(marker.get("id"), str):
        return {}
    return marker


def reconcile_projects(drive_root: Any) -> int:
    """Register projects whose memory store exists but whose registry row is
    missing (e.g. created before the registry shipped). Runs at boot AND on the
    periodic supervisor reconcile tick. A workspace-derived ``proj_<hash>``
    store registers only when a durable task binding OR the store's own
    registry-authored ``.project.json`` marker proves a real room. NEVER
    prunes — durable project dirs outlive any registry accident.
    """
    added = 0
    try:
        projects_root = pathlib.Path(drive_root) / "projects"
        if projects_root.is_dir():
            with _file_write_lock(_registry_path(drive_root)):
                data = _load(drive_root)
                known = {p.get("id") for p in data["projects"]}
                # Keep the store-side marker current for every ACTIVE owner-originated
                # row whose store already exists. Maintained HERE — by the same organ
                # that consumes it — so there is no second producer seam and no mkdir:
                # a file-less room gets its marker on the first tick after its store
                # materializes, and reconcile-originated rows are deliberately
                # excluded so a pre-guard ghost row can never mint recovery evidence
                # for itself.
                for row in data["projects"]:
                    store = projects_root / str(row.get("id") or "")
                    if row.get("lifecycle") == PROJECT_TOMBSTONED:
                        # Convergence for deletion: a marker whose unlink failed at
                        # tombstone time (or predates this line) is retried every
                        # tick, so a stale marker can never outlive its tombstone
                        # into a later registry-loss recovery.
                        try:
                            (store / _PROJECT_MARKER_NAME).unlink(missing_ok=True)
                        except OSError:
                            log.warning("stale marker removal failed for %s", row.get("id"))
                        continue
                    if row.get("lifecycle") != PROJECT_ACTIVE or row.get("origin") == "reconcile":
                        continue
                    if not store.is_dir():
                        continue
                    marker = _read_store_marker(store)
                    if marker.get("id") != row.get("id") or marker.get("name") != row.get("name"):
                        try:
                            atomic_write_json(store / _PROJECT_MARKER_NAME, {
                                "id": row.get("id"),
                                "name": row.get("name"),
                                "origin": row.get("origin"),
                                "created_at": row.get("created_at"),
                            })
                        except OSError:
                            # One unwritable store must not abort the whole reconcile:
                            # the marker retries next tick, registrations still run.
                            log.warning("project marker write failed for %s", row.get("id"),
                                        exc_info=True)
                # A workspace-derived ``proj_<hash>`` store is minted by
                # project_facts for ANY task carrying a workspace path; the
                # directory alone is not evidence that the owner ever created a
                # project room. A durable task binding proves it, and so does the
                # marker above (written only for rows that once really existed).
                # Read the bindings through the cached lens ONCE here — this
                # function also runs on the 300s supervisor reconcile tick, not
                # just at boot. An unreadable bindings file fail-closes to an
                # empty set (every unbound proj_ dir is skipped this pass,
                # idempotently retried on the next tick), and a malformed legacy
                # row — including a non-string ``project_id``, whose str() form
                # could otherwise sanitize into a valid-looking id — is skipped
                # per row rather than coerced or aborting the whole reconcile.
                bound = {
                    sanitize_project_id(row["project_id"])
                    for row in _bindings_lens(drive_root).values()
                    if isinstance(row, dict) and isinstance(row.get("project_id"), str)
                }
                for entry in sorted(projects_root.iterdir()):
                    if not entry.is_dir() or entry.name.startswith("."):
                        continue
                    pid = sanitize_project_id(entry.name)
                    if not pid or pid in known:
                        continue
                    # Named stores register as before. A legitimately created
                    # proj_*-named project passed through create_project and is
                    # already in ``known``, so it never reaches this guard.
                    row_name, row_origin, row_created = pid, "reconcile", ""
                    if pid.startswith("proj_") and pid not in bound:
                        marker = _read_store_marker(entry)
                        if marker.get("id") != pid:
                            continue  # unbound, unmarked: a workspace facts store
                        # Recovery with the room's REAL identity: the pre-guard
                        # behavior resurrected it under the machine name; the
                        # marker restores display name, origin and created_at.
                        row_name = str(marker.get("name") or pid)[:80] or pid
                        row_origin = str(marker.get("origin") or "reconcile") or "reconcile"
                        row_created = str(marker.get("created_at") or "")
                    data["projects"].append({
                        "id": pid,
                        "name": row_name,
                        "chat_id": project_chat_id(pid),
                        "working_dir": "",
                        "origin": row_origin,
                        "created_at": row_created or utc_now_iso(),
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
        })
    return out


__all__ = [
    "PROJECT_ACTIVE",
    "PROJECT_DELETING",
    "PROJECT_NAME_MAX",
    "PROJECT_TOMBSTONED",
    "all_task_bindings",
    "begin_project_deletion",
    "bind_task_to_project",
    "complete_project_deletion",
    "create_project",
    "delete_project",
    "ensure_project_workspace",
    "fail_project_deletion",
    "get_project",
    "get_reserved_project",
    "increment_project_visible_revision",
    "list_projects",
    "list_reserved_projects",
    "list_sidebar_projects",
    "project_binding_for_task",
    "project_chat_for_task",
    "project_thread_note_for_task",
    "project_chat_for_task_tree",
    "project_task_bindings",
    "task_presentation_snapshot",
    "registered_project_chat_ids",
    "reserved_project_chat_ids",
    "projects_summary",
    "reconcile_projects",
    "touch_project",
    "update_project",
]


def project_thread_note_for_task(task: Any) -> str:
    """One-line pointer to the Project thread when a task is project-bound.

    The raw final answer of a bound task lives in the PROJECT room while the
    initiating (Main) chat receives only the task summary — twice in one night
    the owner read that silence as a hung agent. The pointer names where the
    full result lives (v6.70.0); an unbound task gets no extra text."""
    try:
        import pathlib as _pathlib

        from ouroboros.config import DATA_DIR

        chat_id = project_chat_for_task_tree(
            _pathlib.Path(DATA_DIR),
            str(task.get("id") or ""),
            str(task.get("parent_task_id") or ""),
            str(task.get("root_task_id") or ""),
        )
        if not chat_id or int(task.get("chat_id") or 0) == int(chat_id):
            return ""
        name = next(
            (
                str(project.get("name") or "").strip()
                for project in list_projects(_pathlib.Path(DATA_DIR))
                if int(project.get("chat_id") or 0) == int(chat_id)
            ),
            "",
        )
        return f" Full result in the '{name}' project thread." if name else " Full result in the project thread."
    except Exception:
        return ""

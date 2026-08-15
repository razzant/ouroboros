"""Durable registry of owner projects (multi-project, v6.32.0).

A project is a durable context the single agent works in: id + name +
per-project memory (``data/projects/<id>/``) + chat thread (its own positive
``chat_id``) + an OPTIONAL working folder (invisible auto-git under the
durable projects root). File-less research projects are valid. Projects are
NEVER age-pruned; the owner curates by archive/delete.

**Placement (RWS v2 §3.1/§3.3).** A project's WORKING FOLDER may live on another
host. The optional ``placement`` field carries the sealed
:class:`~ouroboros.workspace_ref.SshWorkspaceRef` of a REMOTE project, and it is
stored for remote projects ONLY: a local project's placement IS its
``working_dir``, and persisting the same fact twice is how two authorities start
disagreeing about where a project lives. A row without ``placement`` therefore
reads as local — which is exactly what every pre-RWS row is. Placement lives on
the PROJECT and never on a task because this is the record that carries
``routing_generation``: every placement change advances it (see
:func:`set_project_placement`), and ``supervisor/queue.py`` revalidates that
generation under ``_queue_lock`` before a task becomes PENDING, so work resolved
against a target the owner has since rebound is refused rather than run on the
wrong host. A task that could name its own remote target would be a placement
with no generation to fence.

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
from ouroboros.utils import atomic_write_json, iter_jsonl_objects, read_json_dict, utc_now_iso
from ouroboros.workspace_ref import SshWorkspaceRef, normalize_workspace_ref

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


def _normalized_placement(value: Any) -> Optional[SshWorkspaceRef]:
    """Seal a REMOTE placement for durable storage, or ``None`` when there is none.

    A LOCAL ref is refused rather than silently mirrored: ``working_dir`` already is
    the local placement, so accepting one here would create a second authority for
    the same fact. Anything malformed raises (``normalize_workspace_ref``) — a
    placement this build cannot honor must never be coerced into one it can.
    """
    ref = normalize_workspace_ref(value)
    if ref is None:
        return None
    if not isinstance(ref, SshWorkspaceRef):
        raise ValueError(
            "a local project's placement is its working_dir; only a remote placement is stored"
        )
    return ref


def project_placement(project: Any) -> Optional[SshWorkspaceRef]:
    """The SEALED remote placement of a registry row, or ``None`` for a local project.

    Raises ``ValueError`` LOUDLY for a durable row whose placement this build cannot
    honor. The alternative — reading it as absent — would read a remote project as a
    local one and run its work on the wrong host (the no-silent-fallback invariant).
    """
    raw = project.get("placement") if isinstance(project, dict) else None
    return _normalized_placement(raw)


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


def create_project(
    drive_root: Any,
    project_id: str,
    *,
    name: str = "",
    working_dir: str = "",
    origin: str = "owner",
    placement: Any = None,
) -> Dict[str, Any]:
    """Register (or idempotently return) a project entry.

    ``working_dir`` is optional — file-less projects (research, presentations
    drafted in chat) are first-class. The per-project chat id is derived
    deterministically from the id (one allocator-free SSOT).

    ``placement`` is the sealed REMOTE placement of a project whose folder lives on
    another host (already admitted by ``workspace_admission`` — this registry stores
    placements, it does not validate targets). It is mutually exclusive with
    ``working_dir``: a remote project has no Home folder, so offering one would be a
    Home path standing in for a target the caller never verified.
    """
    pid = sanitize_project_id(project_id)
    if not pid:
        raise ValueError(f"unusable project id: {project_id!r}")
    placement_ref = _normalized_placement(placement)
    if placement_ref is not None and str(working_dir or "").strip():
        raise ValueError("a remote project has no Home working_dir")
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
        if placement_ref is not None:
            entry["placement"] = placement_ref.to_payload()
        data["projects"].append(entry)
        _save(drive_root, data)
        log.info("Project registered: %s (chat_id=%s)", pid, entry["chat_id"])
        return dict(entry)


def update_project(drive_root: Any, project_id: str, **updates: Any) -> Optional[Dict[str, Any]]:
    """Update mutable fields. v6.59.0 adds the additive source-provenance facts:
    ``provenance`` (attached|cloned|genesis|none — how the working_dir came to be),
    ``clone_url`` (historical fact; live git data is always read from .git), and
    ``trusted_at`` (stamped automatically on attach/clone — the notification trust
    model: attaching IS the owner's explicit grant, no second confirmation gate).

    ``placement`` is deliberately NOT updatable here: every placement change must
    advance ``routing_generation``, which is what :func:`set_project_placement` is
    for. A placement that moved through this door would leave the fence pointing at
    a target that no longer exists."""
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
                if key == "working_dir" and entry.get("placement"):
                    # A remote project with a Home working_dir would give every
                    # `working_dir` reader a local folder to prefer over the target.
                    raise ValueError(
                        f"project {pid!r} is remote: rebind its placement instead of "
                        "setting a Home working_dir"
                    )
                if key == "name":
                    value = _validated_name(value, str(entry.get("id") or ""))
                entry[key] = value
            _save(drive_root, data)
            return dict(entry)
    return None


def set_project_placement(
    drive_root: Any,
    project_id: str,
    placement: Any,
    *,
    expected_routing_generation: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """REBIND a project's remote placement and advance its routing generation.

    Separate from :func:`update_project` for one reason: every placement change MUST
    advance ``routing_generation``. That counter is the fence
    ``supervisor/queue.py`` revalidates under ``_queue_lock`` before a task becomes
    PENDING, so a placement that moved without advancing it would let work resolved
    against the previous target be inserted as though nothing had changed.

    ``expected_routing_generation`` is a compare-and-set: two owner rebinds racing
    each other must not interleave silently, and the loser is TOLD it lost
    (``project_routing_generation_changed``) instead of overwriting the winner.

    An identical placement is a no-op that does NOT advance the generation — a
    resubmitted form must not invalidate placements that are still correct. A
    project holding a Home ``working_dir`` is refused: dropping that binding to make
    room for a remote one would discard the owner's folder attachment silently.

    Returns the updated row, ``None`` for an unknown/inactive project, and raises
    ``ValueError`` for a refused rebind.
    """
    pid = sanitize_project_id(project_id)
    if not pid:
        return None
    ref = _normalized_placement(placement)
    if ref is None:
        raise ValueError("set_project_placement requires a remote placement")
    with _file_write_lock(_registry_path(drive_root)):
        data = _load(drive_root)
        for entry in data["projects"]:
            if entry.get("id") != pid or entry.get("lifecycle") != PROJECT_ACTIVE:
                continue
            generation = int(entry.get("routing_generation") or 0)
            if (
                expected_routing_generation is not None
                and generation != int(expected_routing_generation)
            ):
                raise ValueError("project_routing_generation_changed")
            if str(entry.get("working_dir") or "").strip():
                raise ValueError(
                    f"project {pid!r} is bound to a Home folder; a remote placement "
                    "would silently discard that binding"
                )
            try:
                current = project_placement(entry)
            except ValueError:
                # A durable placement this build cannot parse is exactly the row an
                # owner needs to REPAIR, so it must not block its own replacement.
                current = None
            if current == ref:
                return dict(entry)
            entry["placement"] = ref.to_payload()
            entry["routing_generation"] = generation + 1
            _save(drive_root, data)
            log.info(
                "Project placement rebound: %s (routing_generation=%s)",
                pid, entry["routing_generation"],
            )
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
            # The sealed remote placement, or None for a local/file-less project —
            # the sidebar has to be able to say WHERE a project's work will run.
            "placement": project.get("placement") or None,
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
    "project_placement",
    "set_project_placement",
    "project_thread_note_for_task",
    "project_chat_for_task_tree",
    "project_task_bindings",
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

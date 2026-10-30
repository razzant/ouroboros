"""Project registry (data/state/projects.json): id/name/memory/positive chat id, optional folder.
Mind/owner closure is reversible; only owners delete. Tombstones preserve history/data and prevent
rediscovery, never age pruning. Store-marker recovery creates fresh incarnations, not lost folders/tombstones
(PERSISTENCE); rooms share one identity (BIBLE P1)."""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
import pathlib
import threading
from contextlib import contextmanager, nullcontext
from typing import Any, Dict, Iterator, List, Optional

from ouroboros.contracts.chat_id_policy import project_chat_id
from ouroboros.composer_cognition import ensure_project_cognition, main_cognition, project_cognition, project_model_choice
from ouroboros.contracts.schema_versions import with_schema_version
from ouroboros.project_facts import project_display_name, sanitize_project_id
from ouroboros.project_closure import apply_closed, select_projects
from ouroboros.store import StoreBusy, open_store
from ouroboros.store_capture import capture
from ouroboros import utils
from ouroboros.utils import atomic_write_json, json_snapshot, read_json_dict, utc_now_iso, strip_markdown

log = logging.getLogger(__name__)

_REGISTRY_NAME = "projects.json"
# Empty witness touched after each registry save: beside it, a missing registry
# is lost authority, not a fresh install that never had a Project room.
_REGISTRY_WITNESS_NAME = "projects.json.committed"
_BINDINGS_NAME = "project_task_bindings.json"
# Additive schema: unstamped rows are version 0; reconstructed rows may lack
# optional provenance fields, whose defaults must remain compatible.
_REGISTRY_SCHEMA_VERSION = 2
# v6.73.0: project_task_bindings.json gains source_text / origin_absent fields; a copied origin with
# attachments adds source_attachments / source_channel / source_text_placeholder (optional).
_BINDINGS_SCHEMA_VERSION = 1
_LOCK = threading.RLock()

PROJECT_NAME_MAX = 80
PROJECT_ACTIVE = "active"
PROJECT_DELETING = "deleting"
PROJECT_TOMBSTONED = "tombstoned"
PROJECT_LIFECYCLES = frozenset({PROJECT_ACTIVE, PROJECT_DELETING, PROJECT_TOMBSTONED})


@contextmanager
def _file_write_lock(target_path: pathlib.Path, *, timeout_sec: float = 4.0) -> Iterator[None]:
    """Server create/bind and worker journal share a sidecar flock for load→modify→atomic-write;
    nested _LOCK serializes threads. A process-local lock alone permits lost updates."""
    from ouroboros.platform_layer import acquire_exclusive_file_lock, release_exclusive_file_lock

    target_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = target_path.with_name(target_path.name + ".lock")
    fd = acquire_exclusive_file_lock(lock_path, timeout_sec=timeout_sec)
    if fd is None:
        raise TimeoutError(f"projects_registry: could not lock {lock_path} in time")
    try:
        if not _LOCK.acquire(timeout=timeout_sec):
            raise TimeoutError("projects_registry: thread lock busy")
        try:
            yield
        finally:
            _LOCK.release()
    finally:
        release_exclusive_file_lock(lock_path, fd)


def _registry_path(drive_root: Any) -> pathlib.Path:
    return pathlib.Path(drive_root) / "state" / _REGISTRY_NAME


def _registry_witness_path(drive_root: Any) -> pathlib.Path:
    return pathlib.Path(drive_root) / "state" / _REGISTRY_WITNESS_NAME


def _mark_registry_committed(drive_root: Any) -> None:
    """Touch the witness after a commit; a failure only defers it to the next save/reconcile,
    and a registry lost inside that window reads as never committed (disclosed gap)."""
    try:
        _registry_witness_path(drive_root).touch()
    except OSError:
        log.warning("Project registry commit witness was not written", exc_info=True)


def _bindings_path(drive_root: Any) -> pathlib.Path:
    return pathlib.Path(drive_root) / "state" / _BINDINGS_NAME


# Keep the registry facade stable; execution reads have their own strict contract.
from ouroboros.project_admission import _routing_row  # noqa: E402
from ouroboros.project_admission import (  # noqa: E402, F401
    ProjectAdmissionError, _strict_admission_snapshot, display_registry_snapshot,
    project_admission_basis, project_admission_guard, project_admission_view,
    project_scope_admission, task_project_membership, validate_project_admission,
)


def _load(drive_root: Any, *, strict: bool = False,
          project_id: Optional[str] = None) -> Dict[str, Any]:
    # A writer validates global identity and its target, preserving every other
    # raw row. None means a full strict reader; "" selects no row (append/census).
    data = (_strict_admission_snapshot(drive_root, allow_missing=True,
            identity_only=project_id is not None, preserve_raw=project_id is not None)[0] if strict
            else display_registry_snapshot(drive_root))
    for index, row in enumerate(data["projects"]):
        if strict and project_id is not None:
            if row["id"] != project_id:
                continue
            row = data["projects"][index] = _routing_row(row)
        try:
            row["visible_revision"] = max(0, int(row.get("visible_revision") or 0))
        except (TypeError, ValueError):
            row["visible_revision"] = 0
        row["delete_error"] = str(row.get("delete_error") or "")
        row["pinned"] = row.get("pinned") is True
    return data


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
    _mark_registry_committed(drive_root)  # after the commit it attests


_NO_BINDINGS: Dict[str, Any] = {"bindings": {}}  # what a reader of an absent or unreadable store shares


def _load_bindings(drive_root: Any, *, strict: bool = False, fresh: bool = False) -> Dict[str, Any]:
    """Every reader shares one parse per file identity (``json_snapshot``) and never mutates it;
    the one writer reads a ``fresh`` copy. Strict raises on an unreadable or malformed store."""
    data = read_json_dict(_bindings_path(drive_root)) if fresh else json_snapshot(_bindings_path(drive_root),
                                                                                  strict=strict)
    if isinstance(data, dict) and isinstance(data.get("bindings"), dict):
        return data
    if strict and data is not None:
        raise ValueError("Project bindings are unavailable")
    return {"bindings": {}} if fresh else _NO_BINDINGS


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


def _origin_media(drive_root: Any, origin: Any) -> Dict[str, Any]:
    """The attachment refs, ``source`` and placeholder mark the origin's canonical row recorded —
    found by the ref's own ids and verified against the whole ref, never guessed from the text.
    No row, another row or an unreadable chain records nothing; the text copy still stands."""
    ref = origin.get("ref") if isinstance(origin, dict) else None
    if not isinstance(ref, dict) or not isinstance(origin.get("text"), str):
        return {}
    from ouroboros.chat_uploads import stored_refs
    from ouroboros.project_dialogue import entry_matches_source_ref
    from supervisor.message_ingress import accepted_chat_message
    try:
        row = accepted_chat_message(drive_root, int(ref.get("chat_id") or 0), str(ref.get("client_message_id") or ""))
    except (OSError, TypeError, ValueError):
        log.debug("origin attachments unreadable for %r", ref, exc_info=True)
        return {}
    if not row or not row.get("attachments") or not entry_matches_source_ref(row, [ref]):
        return {}
    return {"source_attachments": stored_refs(row["attachments"]), "source_channel": str(row.get("source") or ""),
            **({"source_text_placeholder": True} if row.get("text_placeholder") is True else {})}


def bind_task_to_project(
    drive_root: Any,
    task_id: str,
    project_id: str,
    chat_id: Any = None,
    *,
    origin: Dict[str, Any],
    admission_basis: Optional[dict] = None,
    execution_readmission: bool = False,
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
    if admission_basis is None and get_reserved_project(drive_root, pid) is None:
        create_project(drive_root, pid)
    media = _origin_media(drive_root, origin)  # a chat read: never under the registry's cross-process lock
    # Registry lock linearizes binding with destination deletion.
    with _file_write_lock(_registry_path(drive_root)), \
            project_admission_guard(drive_root, admission_basis) if admission_basis is not None else nullcontext():
        registry_data = _load(drive_root, strict=True, project_id=pid)
        project = next((row for row in registry_data["projects"] if row.get("id") == pid), None)
        if not isinstance(project, dict) or project.get("lifecycle") != PROJECT_ACTIVE:
            lifecycle = project.get("lifecycle") if isinstance(project, dict) else "missing"
            raise ValueError(f"project {pid!r} is {lifecycle}; it cannot accept bindings")
        try:
            resolved_chat = int(chat_id if chat_id is not None else project.get("chat_id"))
        except (TypeError, ValueError):
            resolved_chat = project_chat_id(pid)
        origin_fields = _validated_origin(origin, resolved_chat)
        if "source_text" in origin_fields:  # only a cross-thread origin keeps a copy, and so its media
            origin_fields.update(media)
        row = {
            "task_id": tid,
            "project_id": pid,
            "project_chat_id": resolved_chat,
            "bound_at": utc_now_iso(),
            **({"placement_admission": admission_basis} if execution_readmission else {}),
            **origin_fields,
        }
        with _file_write_lock(_bindings_path(drive_root)):
            data = _load_bindings(drive_root, fresh=True)
            existing = data["bindings"].get(tid)
            if isinstance(existing, dict):
                existing_pid = str(existing.get("project_id") or "")
                if existing_pid == pid:
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
            if "closed" in project:
                apply_closed(project, None)
                _save(drive_root, registry_data)
    touch_project(drive_root, pid)
    return dict(row)


def project_task_bindings(drive_root: Any) -> Dict[str, Dict[str, Any]]:
    """Copy of the immutable task-to-Project bindings for read models (the shared parse)."""
    return {str(task_id): dict(row) for task_id, row in _load_bindings(drive_root)["bindings"].items()
            if isinstance(row, dict)}


def all_task_bindings(drive_root: Any) -> Dict[str, int]:
    """Map task_id -> project chat_id for ALL post-hoc 'Turn into project' bindings:
    the explicit roots (cursor fingerprints, the census). One bounded read; no
    per-row lock (atomic writes guarantee complete reads)."""
    return {tid: int(row["chat_id"]) for tid, row in all_task_project_bindings(drive_root).items()}


def all_task_project_bindings(drive_root: Any, *, strict: bool = False) -> Dict[str, Dict[str, Any]]:
    """Map task_id -> {project_id, chat_id} for ALL post-hoc 'Turn into project'
    bindings. Richer than all_task_bindings (chat-id only): the UI uses project_id
    to turn a bound main-chat card into a pointer that opens the project panel
    (F4), not merely to suppress the stray convert button (P2). Strict snapshot
    readers propagate source failures so absence never impersonates completeness."""
    out: Dict[str, Dict[str, Any]] = {}
    try:
        names = {row["id"]: row.get("name") or row["id"] for row in _load(drive_root)["projects"]}
        bindings = _load_bindings(drive_root, strict=strict)["bindings"]
        for tid, row in bindings.items():
            if not isinstance(row, dict):
                continue
            pid = str(row.get("project_id") or "").strip()
            try:
                cid = int(row.get("project_chat_id") or 0)
            except (TypeError, ValueError):
                cid = 0
            if pid and cid:
                out[str(tid)] = {"project_id": pid, "chat_id": cid, "project_name": str(names.get(pid) or pid)}
    except Exception:
        if strict:
            raise
        log.debug("all_task_project_bindings failed", exc_info=True)
    return out


_PROCESS_STAMP = uuid.uuid4().hex[:8]  # this process's code shapes the map it answers


def bindings_revision(drive_root: Any) -> str:
    """An opaque stamp of what ``all_task_project_bindings`` answers in this process: the file
    identities of the bindings and of the registry naming their projects. ``""`` when no stamp can
    prove the bytes unchanged: unreadable, or changed within a clock tick (``utils._RACY_NS``)."""
    started, stamps = time.time_ns(), [_PROCESS_STAMP]
    for path in (_bindings_path(drive_root), _registry_path(drive_root)):
        try:
            info = path.stat()
        except FileNotFoundError:
            stamps.append(None)
            continue
        except OSError:
            return ""
        if max(info.st_mtime_ns, info.st_ctime_ns) >= started - utils._RACY_NS:
            return ""
        stamps.append(utils.file_identity(info))
    return hashlib.sha256(repr(stamps).encode("utf-8")).hexdigest()[:16]


def project_binding_for_task(drive_root: Any, task_id: str, *, strict: bool = False,
                             bindings_snapshot: Optional[dict] = None) -> Optional[Dict[str, Any]]:
    tid = str(task_id or "").strip()
    if not tid:
        return None
    # Read needs no lock: atomic_write_json renames into place, so a reader
    # always sees a complete (old or new) bindings file, never a torn one.
    bindings = (_load_bindings(drive_root, strict=strict)["bindings"]
                if bindings_snapshot is None else bindings_snapshot)
    row = bindings.get(tid)
    if strict and tid in bindings and (not isinstance(row, dict) or not isinstance(row.get("project_id"), str)
            or not row["project_id"] or sanitize_project_id(row["project_id"]) != row["project_id"]):
        raise ValueError("Project binding is unavailable")
    return dict(row) if isinstance(row, dict) else None


def project_chat_for_task(drive_root: Any, task_id: str) -> int:
    row = project_binding_for_task(drive_root, task_id)
    if not row:
        return 0
    try:
        return int(row.get("project_chat_id") or 0)
    except (TypeError, ValueError):
        return 0


def project_id_for_task(drive_root: Any, task_id: str, *, strict: bool = False) -> str:
    """Project a task is DURABLY bound to, "" when it is bound to nothing.

    The binding is the ONE truth about a task's project (owner decision B4=A):
    ``task["project_id"]`` and a worker's in-memory ``ctx.project_id`` are
    copies that a mid-run conversion never reaches. Reads the one shared parse
    the live addressing seam already uses, so a hot-path caller pays no extra
    parse. ``strict=True`` RAISES on an unreadable store instead
    of reporting "unbound", for the one caller that authorizes creating a
    project and must not do that blind.
    """
    tid = str(task_id or "").strip()
    if not tid:
        return ""
    row = _load_bindings(drive_root, strict=strict)["bindings"].get(tid)
    return str(row.get("project_id") or "").strip() if isinstance(row, dict) else ""


def origin_key(ref: Any) -> Optional[tuple]:
    """Identity of an owner message BY VALUE: ``(chat_id, client_message_id)``.

    ``ts``/``text_sha256`` are deliberately NOT part of the key. The binding
    validated them once at write time (``_validated_origin``), and a later
    reader must never re-derive identity from content (DEVELOPMENT.md
    anti-pattern). ``None`` for a missing/malformed ref, so a caller can skip
    the lookup entirely instead of matching on an empty key.
    """
    if not isinstance(ref, dict):
        return None
    message = str(ref.get("client_message_id") or "").strip()
    if not message or ref.get("chat_id") in (None, ""):
        return None
    try:
        return (int(ref["chat_id"]), message)
    except (TypeError, ValueError):
        return None


_ORIGIN_CLAIM_LOCK = threading.RLock()


def origin_claim_lock() -> Any:
    """Serialize mind/button placement from the task binding read through commit.

    Naming stays outside the lock. Shared message origin is provenance only;
    independent works never acquire each other's room through this serializer.
    """
    return _ORIGIN_CLAIM_LOCK


def _live_task_ids() -> set:
    """Best-effort ids of tasks that are running right now: queue roots/children and
    in-flight direct turns. Used ONLY to break a tie between several projects bound
    to one origin (legacy state), so the common path never touches the supervisor."""
    live: set = set()
    try:
        from supervisor.queue import _queue_lock
        from supervisor.workers import PENDING, RUNNING

        with _queue_lock:
            live.update(str(tid) for tid in RUNNING)
            live.update(
                str(row.get("id") or "") for row in (PENDING or ()) if isinstance(row, dict)
            )
    except Exception:
        log.debug("_live_task_ids: queue read failed", exc_info=True)
    try:
        from supervisor.active_activity import get_direct_activity_registry

        live.update(str(entry.activity_id) for entry in get_direct_activity_registry().actors())
    except Exception:
        log.debug("_live_task_ids: direct activity read failed", exc_info=True)
    live.discard("")
    return live


def project_id_for_origin(drive_root: Any, origin_ref: Any, *, strict: bool = False,
                          include_inactive: bool = False, bindings_snapshot: Optional[dict] = None) -> str:
    """Project bound to ANY task whose binding names this owner-message origin, else "".

    One owner message can spawn several task ids (the direct turn that received
    it, a root it promotes, a retry of that root), and each of them used to be a
    separately convertible unit — which is how ONE message minted TWO Projects.
    The origin is the identity of that work: it is captured at ingress, copied
    by value onto every task the message spawns, and already stored in the
    binding's ``source_ref``, so this reverse lookup needs no new field. A
    malformed ref resolves to "" and never aliases another message.

    Reads the same shared parse ``project_id_for_task`` uses;
    ``strict=True`` RAISES on an unreadable store, for the callers
    that authorize creating a project. A binding whose project is no longer
    ACTIVE does not count, unless admission requests ``include_inactive`` to
    preserve positive membership evidence even after a room closes/disappears.
    LEGACY state can name SEVERAL active projects for one origin (the measured
    incident left exactly that): prefer the one whose task is still LIVE, else the
    LATEST binding — where the work actually continued — and disclose the choice
    durably. Never raises on ambiguity; a refusal would either block the owner or
    re-mint the duplicate it is meant to end.
    """
    key = origin_key(origin_ref)
    if key is None:
        return ""
    bindings = bindings_snapshot
    if bindings is None:
        bindings = _load_bindings(drive_root, strict=strict)["bindings"]
    candidates = []
    for task_id, row in bindings.items():
        if not isinstance(row, dict) or origin_key(row.get("source_ref")) != key:
            continue
        pid = row.get("project_id")
        # Validate matching evidence before display coercion can alias an ID or
        # turn corrupt positive membership into an apparently unbound origin.
        if strict and (not isinstance(pid, str) or not pid or sanitize_project_id(pid) != pid):
            raise ValueError("Project origin binding is unavailable")
        pid = str(pid or "").strip()
        if pid:
            candidates.append((str(row.get("bound_at") or ""), str(task_id), pid))
    candidates.sort()
    if not candidates:
        return ""  # the common case: no binding names this message, so no registry read
    if not include_inactive:
        if strict:
            ids = {row[2] for row in candidates}
            projects = (_routing_row(row) for row in _load(drive_root, strict=True, project_id="")["projects"]
                        if row["id"] in ids)
        else:
            projects = list_projects(drive_root)
        active = {project["id"] for project in projects if project["lifecycle"] == PROJECT_ACTIVE}
        candidates = [row for row in candidates if row[2] in active]
    if not candidates:
        return ""
    chosen = candidates[-1][2]
    if len({row[2] for row in candidates}) > 1:
        live = _live_task_ids()
        chosen = next(
            (row[2] for row in reversed(candidates) if row[1] in live), candidates[-1][2],
        )
        _emit_origin_ambiguous(drive_root, key, candidates, chosen)
    return chosen


_DISCLOSED_AMBIGUOUS_ORIGINS: set = set()


def _emit_origin_ambiguous(drive_root: Any, key: tuple, candidates: list, chosen: str) -> None:
    """Durable disclosure (P1) that one owner message names several projects, and
    which one the adopt chose. Best-effort; never raises.

    ONCE per (store, origin, choice) per process: legacy state is read on every
    promote tool call, every promote admission and every convert click, and a fact
    that has not changed is not news — a row per lookup would bury the disclosure in
    its own repetitions. A restart discloses once more, and a CHANGED choice (the
    bound task went live, or a candidate's project was deleted) is a new fact and is
    always written."""
    disclosure = (str(drive_root), key, chosen)
    if disclosure in _DISCLOSED_AMBIGUOUS_ORIGINS:
        return
    try:
        from ouroboros.utils import append_jsonl

        # Marked only once the row is durably appended: a failed append must not
        # silence every later disclosure of the same fact in this process.
        written = append_jsonl(pathlib.Path(drive_root) / "logs" / "events.jsonl", {
            "ts": utc_now_iso(),
            "type": "project_origin_ambiguous",
            "origin": {"chat_id": key[0], "client_message_id": key[1]},
            "candidates": [
                {"task_id": task_id, "project_id": project_id, "bound_at": bound_at}
                for bound_at, task_id, project_id in candidates
            ],
            "chosen": chosen,
        })
        if written:
            _DISCLOSED_AMBIGUOUS_ORIGINS.add(disclosure)
    except Exception:
        log.debug("project_origin_ambiguous row failed", exc_info=True)


_TASK_LINEAGE: Dict[tuple, Dict[str, Any]] = {}
_TASK_LINEAGE_LIMIT = 50_000  # a memory bound, not a policy: the facts are re-read on the next sight
_ROOM_ANSWERS: Dict[str, tuple] = {}
# One capture (``store_capture``); ``chat_id`` as text or, from a JSON real, its integer part (what ``int()`` gave).
_LINEAGE_SELECT = ("SELECT t.task_id, t.parent_task_id, t.root_task_id, t.delegation_role, CASE json_type(b.body, "
                   "'$.chat_id') WHEN 'real' THEN CAST(json_extract(b.body, '$.chat_id') AS INTEGER) ELSE '' || "
                   "json_extract(b.body, '$.chat_id') END AS chat_id FROM tasks t LEFT JOIN task_bodies b ON "
                   "b.task_id = t.task_id WHERE t.task_id IN (SELECT value FROM json_each(?))")


def _read_lineage(drive_root: Any, task_ids: Any, *, strict: bool = False) -> Optional[Dict[str, Dict[str, Any]]]:
    """ONE indexed read of the lineage facts of ``task_ids`` (parent, root, role and the
    admitted ``chat_id``), remembered per task id for the process (an existing task's
    lineage never changes); an id the store does not hold is absent from the answer
    (admission creates it later); ``None`` when the store cannot answer right now, except
    that a ``strict`` reader keeps positive contention typed (``StoreBusy``) for its
    acquisition owner to wait out: contention is not unknown lineage."""
    try:
        with open_store(drive_root, write=False) as txn:
            captured = capture(txn.conn, _LINEAGE_SELECT, (json.dumps(sorted(task_ids)),),
                               ("task_id", "parent_task_id", "root_task_id", "delegation_role", "chat_id"))
        records = captured.tuples()
    except StoreBusy as exc:
        if strict and exc.reason == "contention":
            raise
        log.debug("task lineage read failed", exc_info=True)
        return None
    except Exception:
        log.debug("task lineage read failed", exc_info=True)
        return None
    if len(_TASK_LINEAGE) >= _TASK_LINEAGE_LIMIT:
        _TASK_LINEAGE.clear()
    facts = {}
    for record in records:
        try:
            chat = int(record[4]) if record[4] not in (None, "") else None
        except (TypeError, ValueError):
            chat = None
        facts[str(record[0])] = {"parent_task_id": str(record[1] or ""), "root_task_id": str(record[2] or ""),
                                 "delegation_role": str(record[3] or ""), "chat_id": chat}
    for tid, row in facts.items():
        _TASK_LINEAGE[(str(drive_root), tid)] = row
    return facts


class RoomResolver:
    """One rule for live/history/cognition: nearest explicit binding on task→parent→…,
    then root; else own admitted registered Project chat (never project_id), else 0
    preserves the caller's address. Caller/store lineage is shared. Answers cache per
    bindings snapshot: conversion invalidates, commit does not; unavailable ancestors
    retry, absent ids read once per resolver. Strict publication uses fresh bindings
    and propagates unreadable authority instead of pinning a guessed room."""

    __slots__ = ("root", "bindings", "answers", "misses", "_projects", "strict")

    def __init__(self, drive_root: Any, *, strict: bool = False) -> None:
        self.root, self.strict = drive_root, strict
        self.bindings = _load_bindings(drive_root, strict=strict)["bindings"]  # one ``stat`` per resolver
        key = str(_bindings_path(drive_root))
        cached = None if strict else _ROOM_ANSWERS.get(key)
        if cached is None or cached[0] is not self.bindings:
            cached = (self.bindings, {})
            if not strict:
                _ROOM_ANSWERS[key] = cached
        self.answers: Dict[str, int] = cached[1]
        self.misses: set = set()
        self._projects: Optional[frozenset] = None

    def _facts(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Cached or read facts; ``{}`` for a known miss; ``None`` when unavailable now."""
        facts = _TASK_LINEAGE.get((str(self.root), task_id))
        if facts is None and task_id not in self.misses:
            read = _read_lineage(self.root, [task_id], strict=self.strict)
            if read is None:
                if self.strict:
                    raise ValueError("project_source_unknown")
                return None
            facts = read.get(task_id)
            if facts is None:
                self.misses.add(task_id)
        return facts or {}

    def lineage(self, task_id: Any) -> Dict[str, Any]:
        """The task's facts (``{}`` when unknown or unavailable now): what a reader anchors by."""
        tid = str(task_id or "").strip()
        return (self._facts(tid) if tid else None) or {}

    def _bound(self, task_id: str) -> int:
        row = project_binding_for_task(self.root, task_id, strict=self.strict, bindings_snapshot=self.bindings)
        if self.strict and row:
            project = get_reserved_project(self.root, row["project_id"], strict=True)
            if not project or type(project.get("chat_id")) is not int:
                raise ValueError("project_source_unknown")
            return project["chat_id"]
        try:
            return int(row.get("project_chat_id") or 0) if isinstance(row, dict) else 0
        except (TypeError, ValueError):
            return 0

    def __call__(self, task_id: Any, parent_task_id: Any = "", root_task_id: Any = "") -> int:
        tid = str(task_id or "").strip()
        if not tid:
            return 0
        answer = self.answers.get(tid)
        if answer is not None:
            return answer
        reads: List[Optional[Dict[str, Any]]] = []  # ``None``: an ancestor the store could not answer
        chain: List[str] = []
        cur, parent, chat = tid, str(parent_task_id or "").strip(), 0
        while cur and cur not in chain:
            chain.append(cur)
            if chat := self._bound(cur):
                break
            if not parent:
                reads.append(self._facts(cur))
                parent = (reads[-1] or {}).get("parent_task_id", "")
            cur, parent = parent, ""
        own = {} if chat else self._facts(tid)
        reads.append(own)
        own = own or {}
        if not chat:
            root = str(root_task_id or "").strip() or own.get("root_task_id", "")
            chat = self._bound(root) if root and root not in chain else 0
            if not chat and own.get("chat_id"):
                if self._projects is None:
                    self._projects = (frozenset(reserved_project_chat_ids(self.root, strict=True)) if self.strict
                                      else project_thread_chat_ids(self.root))
                chat = own["chat_id"] if own["chat_id"] in self._projects else 0
        if (chat or own) and None not in reads:  # an incomplete ancestry read is never memoized
            self.answers[tid] = chat
        return chat

    def warm(self, task_ids: Any) -> bool:
        """Read, ONE transaction per generation, the lineage the rule then walks one id at a time (a page
        names dozens): every unseen id and the parents and roots of the rows read now, and from a
        remembered task its parent while the walk goes on there (neither is bound to a Project; a root
        is only asked for its binding); ``False`` when the store could not answer them all now."""
        wanted, seen = {str(tid) for tid in task_ids if tid}, set()
        while wanted:
            seen |= wanted
            unseen = {tid for tid in wanted if (str(self.root), tid) not in _TASK_LINEAGE} - self.misses
            read = _read_lineage(self.root, unseen) if unseen else {}
            if read is None:
                return False
            self.misses.update(unseen - read.keys())
            fresh = {row[key] for row in read.values() for key in ("parent_task_id", "root_task_id") if row[key]}
            walked = {row["parent_task_id"] for tid in wanted - unseen if not self._ends_walk(tid)
                      and (row := _TASK_LINEAGE.get((str(self.root), tid))) and row["parent_task_id"]}
            wanted = (fresh | {tid for tid in walked if not self._ends_walk(tid)}) - seen
        return True

    def _ends_walk(self, task_id: str) -> bool:
        """The rule's walk ends at a task bound to a Project; a binding that cannot be read ends nothing here."""
        try:
            return bool(self._bound(task_id))
        except Exception:
            return False


def project_chat_for_task_tree(
    drive_root: Any, task_id: Any, parent_task_id: Any = "", root_task_id: Any = ""
) -> int:
    """The room rule (``RoomResolver``) under its live-addressing name: 0 when none; never raises."""
    try:
        return RoomResolver(drive_root)(task_id, parent_task_id, root_task_id)
    except Exception:
        log.debug("project_chat_for_task_tree failed", exc_info=True)
        return 0


def list_reserved_projects(drive_root: Any, *, strict: bool = False) -> List[Dict[str, Any]]:
    """Raw registry reservations, including frozen pre-cutover revisions for the importer."""
    with _LOCK:
        projects = _load(drive_root, strict=strict)["projects"]
    return sorted(
        projects,
        key=lambda p: str(p.get("last_active_at") or p.get("updated_at") or p.get("created_at") or ""),
        reverse=True,
    )


def list_projects(drive_root: Any, *, strict: bool = False,
                  closed: Optional[bool] = None, oldest_first: bool = False) -> List[Dict[str, Any]]:
    """Active, routable Projects; optional closure filter and oldest activity first."""
    from ouroboros import dialogue_store as ds

    projects = [
        project for project in list_reserved_projects(drive_root, strict=strict)
        if project.get("lifecycle") == PROJECT_ACTIVE
    ]
    facts = {room["chat_id"]: room for room in ds.rooms(drive_root)} if projects else {}
    return select_projects((_with_dialogue_facts(project, facts.get(project["chat_id"])) for project in projects),
                           closed=closed, oldest_first=oldest_first)


def list_sidebar_projects(drive_root: Any) -> List[Dict[str, Any]]:
    """Projects visible while active or while deletion is quiescing."""
    return [
        project for project in list_reserved_projects(drive_root)
        if project.get("lifecycle") in {PROJECT_ACTIVE, PROJECT_DELETING}
    ]


def reserved_project_chat_ids(drive_root: Any, *, strict: bool = False) -> set:
    """The set of chat_ids reserved by every Project lifecycle state.

    The TRUTH source for "is this chat a project thread" — a bare numeric range
    cannot disambiguate from large external-transport (e.g. Telegram) chat ids,
    so routing/history/UI classify by registry membership instead.

    NOT an isolation boundary (full project awareness, v6.32.0): the one identity
    sees ALL threads in its unified memory. This classifier drives (a) the UI
    history/fan-out partition that organizes threads into panels, (b) message
    routing, and (c) the project TASK's FOCUSED passive context (build_recent_
    sections shows the task its own thread). ``strict=True`` RAISES on an
    unreadable registry instead of answering "no Projects".
    """
    try:
        projects = (_strict_admission_snapshot(drive_root, allow_missing=True, identity_only=True)[0]["projects"]
                    if strict else list_reserved_projects(drive_root))
        return {project["chat_id"] for project in projects} - {0}
    except Exception:
        if strict:
            raise
        log.debug("reserved_project_chat_ids failed", exc_info=True)
    return set()


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


def get_project(drive_root: Any, project_id: str, *, strict: bool = False) -> Optional[Dict[str, Any]]:
    project = get_reserved_project(drive_root, project_id, strict=strict)
    if not project or project.get("lifecycle") != PROJECT_ACTIVE:
        return None
    from ouroboros import dialogue_store as ds

    return _with_dialogue_facts(project, ds.room(drive_root, project["chat_id"]))


def get_reserved_project(drive_root: Any, project_id: str, *, strict: bool = False) -> Optional[Dict[str, Any]]:
    """Lookup irrespective of lifecycle (history/recovery only)."""
    pid = sanitize_project_id(project_id)
    if not pid:
        return None
    rows = (_load(drive_root, strict=True, project_id=pid)["projects"] if strict
            else list_reserved_projects(drive_root))
    return next((dict(project) for project in rows if project.get("id") == pid), None)



def _bounded_presentation_name(value: Any, *, fallback: str = "") -> str:
    name = " ".join(str(value or "").split()) or str(fallback or "")
    return name if len(name) <= PROJECT_NAME_MAX else name[: PROJECT_NAME_MAX - 1].rstrip() + "…"


def task_presentation_name(*sources: Any) -> str:
    """Plain work title from structured task/result rows, without storage reads.

    A title or coined name is shown whole; a bare request shows its first line, the
    one derivation admission gives every run (``project_naming.admission_names``)."""
    from ouroboros.project_naming import clean_model_title

    sources = [row for row in sources if isinstance(row, dict)]
    for field in ("title", "suggested_name", "objective", "description"):
        for source in sources:
            # Strip markdown while the lines remain: the first line is the request's name.
            text = strip_markdown(str(source.get(field) or ""))
            name = (_bounded_presentation_name(text) if field in ("title", "suggested_name")
                    else clean_model_title(text, max_len=PROJECT_NAME_MAX))
            if name:
                return name
    return "Task"


def task_presentation_snapshot(drive_root: Any, task_id: str, *, task: Any = None,
                               result: Any = None, project_id: str = "", strict: bool = False) -> Dict[str, Any]:
    tid = str(task_id or "").strip()
    sources = [row for row in (task, result) if isinstance(row, dict)]
    if tid and not strict:
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
        binding = project_binding_for_task(drive_root, tid, strict=strict) or {}
        pid = str(binding.get("project_id") or "").strip()
    pname = ""
    registered = False
    if pid:
        # ONE registry read serves both the display name and the additive
        # ``project_routable`` fact: a workspace-derived proj_<hash> is
        # project-SCOPED without having a room, and a producer that announces it
        # would point the owner at a project that does not exist.
        project = get_reserved_project(drive_root, pid, strict=strict) or {}
        # ROUTABLE, not merely reserved: list_reserved_projects deliberately
        # includes deleting/tombstoned history reservations, and those have no
        # room left to open.
        registered = str(project.get("lifecycle") or "") == PROJECT_ACTIVE
        pname = _bounded_presentation_name(project_display_name(project.get("name"), pid))
    if pid and not pname:
        pname = "Project"
    task_name = task_presentation_name(*sources)
    label = f"{pname} › {task_name}" if pname else task_name
    # One name, said once: a task whose own name IS the project name renders
    # "Launch › Launch", which reads as two different things. Exact equality
    # only — a prefix rule would fold "Art" into "Arthur".
    if pname and task_name == pname:
        label = task_name
    return {"project_id": pid, "project_name": pname, "task_id": tid,
            "project_routable": registered, "task_name": task_name,
            "target_label": label}


# A Project root's final answer rides Main's completion row so Main can show it as
# an ordinary Ouroboros message (DESIGN "Project completion mirror"). Above this
# size the row stays a pointer: nothing is ever cut host-side (BIBLE P1).
MIRRORED_ANSWER_MAX_CHARS = 32000
_TERMINAL_MIRROR_PHASES = frozenset({"done", "warn", "error", "cancelled"})


def mirrored_answer(result: Any, phase: str) -> Dict[str, str]:
    """The typed key Main's completion row carries, or nothing.

    Present only for a MODEL-AUTHORED final answer of a settled task: host
    salvage, internal quiet results and host notices are not published answers; a row sent while the
    task still reads ``working`` is frozen by the outbox and must stay a pointer.
    """
    from ouroboros.task_finalization import TERMINAL_ORIGIN_MODEL_FINAL

    row = result if isinstance(result, dict) else {}
    answer = row["result"].strip() if isinstance(row.get("result"), str) else ""
    if (str(row.get("terminal_origin") or "") != TERMINAL_ORIGIN_MODEL_FINAL
            or row.get("wake_silent") is True
            or str(phase or "") not in _TERMINAL_MIRROR_PHASES
            or not answer or len(answer) > MIRRORED_ANSWER_MAX_CHARS):
        return {}
    return {"completion_answer": answer}


def create_project(
    drive_root: Any,
    project_id: str,
    *,
    name: str = "",
    working_dir: str = "",
    origin: str = "owner",
    admission_basis: Optional[dict] = None,
) -> Dict[str, Any]:
    """Register an active room with deterministic chat id and an optional folder.
    The unpersisted ``created`` flag distinguishes this registration from replay."""
    pid = sanitize_project_id(project_id)
    if not pid:
        raise ValueError(f"unusable project id: {project_id!r}")
    cognition = main_cognition()
    ensure_project_cognition(drive_root, baseline=cognition)
    with _file_write_lock(_registry_path(drive_root)), \
            project_admission_guard(drive_root, admission_basis) if admission_basis is not None else nullcontext():
        data = _load(drive_root, strict=True, project_id=pid)
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
            "cognition": cognition,
            "origin": str(origin or "owner"),
            "created_at": utc_now_iso(),
            "last_active_at": utc_now_iso(),
            "lifecycle": PROJECT_ACTIVE,
            "routing_generation": 0,
            "routing_incarnation": uuid.uuid4().hex,
            "visible_revision": 0,
            "pinned": False,
            "delete_error": "",
        }
        data["projects"].append(entry)
        _save(drive_root, data)
        log.info("Project registered: %s (chat_id=%s)", pid, entry["chat_id"])
        return {**entry, "created": True}


def update_project(
    drive_root: Any, project_id: str, *, only_if_empty: tuple = (),
    admission_basis: Optional[dict] = None, return_changed: bool = False, **updates: Any,
) -> Optional[Dict[str, Any] | tuple[Dict[str, Any], bool]]:
    """Update metadata, including ``closed={at, by, note}`` or None to reopen.
    Provenance/clone_url/trusted_at are historical attach facts, not live Git data.
    ``only_if_empty`` compares under the writer lock; ``return_changed`` also returns whether it wrote."""
    pid = sanitize_project_id(project_id)
    if not pid:
        return None
    allowed = {
        "name", "pinned", "closed", "working_dir", "last_active_at", "provenance", "clone_url", "trusted_at",
        # Retained legacy activity fact; current activity comes from dialogue rooms.
        "thread_activity_seen",
        # Durable pointer to the project's newest finalized task result
        # (stamped by record_task_finalization; read first by
        # server._latest_project_task_result before its bounded scan).
        "last_task_result_id",
    }
    with _file_write_lock(_registry_path(drive_root)):
        if admission_basis is not None:
            with project_admission_guard(drive_root, admission_basis):
                pass  # the writer lock keeps this comparison valid through _save
        data = _load(drive_root, strict=True, project_id=pid)
        for entry in data["projects"]:
            if entry.get("id") != pid or entry.get("lifecycle") != PROJECT_ACTIVE:
                continue
            original = dict(entry)
            for key, value in updates.items():
                if key not in allowed or (key in only_if_empty and str(entry.get(key) or "").strip()):
                    continue
                if key == "name":
                    value = _validated_name(value, str(entry.get("id") or ""))
                if key == "pinned" and type(value) is not bool:
                    raise ValueError("pinned must be a boolean")
                if key == "working_dir" and value != entry.get(key):
                    entry["routing_generation"] = int(entry.get("routing_generation") or 0) + 1
                if key == "closed":
                    apply_closed(entry, value)
                else:
                    entry[key] = value
            if entry != original:
                _save(drive_root, data)
            return (dict(entry), entry != original) if return_changed else dict(entry)
    return None


def begin_project_deletion(drive_root: Any, project_id: str) -> Optional[Dict[str, Any]]:
    """Close admission/routing before the supervisor cancels the live subtree."""
    pid = sanitize_project_id(project_id)
    if not pid:
        return None
    with _file_write_lock(_registry_path(drive_root)):
        data = _load(drive_root, strict=True, project_id=pid)
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
        data = _load(drive_root, strict=True, project_id=pid)
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
        data = _load(drive_root, strict=True, project_id=pid)
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
        ensure_project_cognition(drive_root)
        cognition = main_cognition()
        if _registry_path(drive_root).is_file():
            _mark_registry_committed(drive_root)  # a registry saved before its witness existed
        projects_root = pathlib.Path(drive_root) / "projects"
        if projects_root.is_dir():
            with _file_write_lock(_registry_path(drive_root)):
                try:
                    data = _load(drive_root, strict=True, project_id="")
                except FileNotFoundError:
                    # Reconstruction owns absence, unlike admission. Recovered
                    # rooms receive NEW incarnations and never authorize old work.
                    data = {"projects": []}
                known = {p.get("id") for p in data["projects"]}
                # Keep the store-side marker current for every ACTIVE owner-originated
                # row whose store already exists. Maintained HERE — by the same organ
                # that consumes it — so there is no second producer seam and no mkdir:
                # a file-less room gets its marker on the first tick after its store
                # materializes, and reconcile-originated rows are deliberately
                # excluded so a pre-guard ghost row can never mint recovery evidence
                # for itself.
                for raw in data["projects"]:
                    try:
                        row = _routing_row(raw)
                    except ValueError:
                        continue  # retain unknown routing; do not maintain its marker
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
                    if any(marker.get(key) != row.get(key) for key in ("id", "name", "cognition")):
                        try:
                            atomic_write_json(store / _PROJECT_MARKER_NAME, {
                                "id": row.get("id"),
                                "name": row.get("name"),
                                "origin": row.get("origin"),
                                "created_at": row.get("created_at"),
                                "cognition": row.get("cognition"),
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
                # Read the bindings through the shared parse ONCE here — this
                # function also runs on the 300s supervisor reconcile tick, not
                # just at boot. An unreadable bindings file fail-closes to an
                # empty set (every unbound proj_ dir is skipped this pass,
                # idempotently retried on the next tick), and a malformed legacy
                # row — including a non-string ``project_id``, whose str() form
                # could otherwise sanitize into a valid-looking id — is skipped
                # per row rather than coerced or aborting the whole reconcile.
                bound = {
                    sanitize_project_id(row["project_id"])
                    for row in _load_bindings(drive_root)["bindings"].values()
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
                    marker = _read_store_marker(entry)
                    if pid.startswith("proj_") and pid not in bound:
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
                        "cognition": project_cognition(marker, cognition),
                        "origin": row_origin,
                        "created_at": row_created or utc_now_iso(),
                        "last_active_at": utc_now_iso(),
                        "lifecycle": PROJECT_ACTIVE,
                        "routing_generation": 0,
                        "routing_incarnation": uuid.uuid4().hex,
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
    return added


def ensure_project_workspace(drive_root: Any, project_id: str, repo_dir: Any,
                             *, return_project: bool = False, admission_basis: Optional[dict] = None) -> Any:
    """Choose the existing folder or provision only while the folder is empty.

    Preparation callers can consume the exact chosen row with return_project;
    reading again after provisioning would lose an intervening rebind/ABA.
    """
    entry = create_project(drive_root, project_id, admission_basis=admission_basis)
    if not str(entry.get("working_dir") or "").strip():
        try:
            from ouroboros.subagent_worktrees import provision_genesis_project

            handle = provision_genesis_project(
                repo_dir=repo_dir, task_id=f"project_{entry['id']}", data_dir=drive_root,
                dir_name=str(entry.get("name") or ""),
            )
            entry = update_project(drive_root, entry["id"], working_dir=str(handle.path),
                                   only_if_empty=("working_dir",),
                                   admission_basis=project_admission_basis(entry["id"], entry))
        except Exception:
            log.warning("Project workspace provisioning failed for %s", project_id, exc_info=True)
            entry = None
    if return_project:
        return entry
    return str((entry or {}).get("working_dir") or "")


def _with_dialogue_facts(project: Dict[str, Any], room: Optional[dict]) -> Dict[str, Any]:
    facts = room or {}
    return {**project, "visible_revision": int(facts.get("revision") or 0),
            "last_active_at": max(str(project.get("last_active_at") or ""), str(facts.get("activity_at") or ""))}


def projects_summary(drive_root: Any, *, limit: int = 50, chat_id: Any = None,
                     include_pinned: bool = False, closed: Optional[bool] = None,
                     oldest_first: bool = False, start: int = 0) -> List[Dict[str, Any]]:
    """Sidebar or strict addressed room, with unread/activity from dialogue room facts.

    Registry reads retain lifecycle and binding authority. An unproved inactive room
    requires complete import; a pending, failed or invalid interval is never inactivity.
    """
    from ouroboros import dialogue_store as ds

    out: List[Dict[str, Any]] = []
    cognition = main_cognition()
    bound = {str(row.get("project_id") or "") for row in _load_bindings(drive_root)["bindings"].values()
             if isinstance(row, dict)}
    if chat_id is not None:
        address = int(chat_id)
        projects = [row for row in list_reserved_projects(drive_root, strict=True)
                    if row.get("chat_id") == address and row.get("lifecycle") == PROJECT_ACTIVE]
        if len(projects) > 1:
            raise ValueError("Project chat identity is ambiguous")
        facts = {address: ds.room(drive_root, address)} if projects else {}
    else:
        projects = [p for p in list_sidebar_projects(drive_root)
                    if closed is None or p.get("lifecycle") == PROJECT_ACTIVE]
        facts = {room["chat_id"]: room for room in ds.rooms(drive_root)} if projects else {}
    projects = [_with_dialogue_facts(project, facts.get(project["chat_id"])) for project in projects]
    projects = select_projects(projects, closed=closed, oldest_first=oldest_first)
    if chat_id is None and include_pinned:
        # Pins remain reachable without starving the ordinary recent-room slice.
        projects = ([p for p in projects if p.get("pinned") is True]
                    + [p for p in projects if p.get("pinned") is not True][:max(1, int(limit))])
    elif chat_id is None:
        start = max(0, int(start))
        projects = projects[start:start + max(1, int(limit))]
    absence_covered = False
    for project in projects:
        room = facts.get(project["chat_id"]) or {}
        active = bool(project.get("origin") == "owner_ui" or project["id"] in bound
                      or project.get("thread_activity_seen") or room.get("revision")
                      or room.get("activity_at") or room.get("last_seq") is not None)
        if not active and room:
            # A progress row without a timestamp still proves activity. A bare read
            # marker can create an empty room too, so presence of the room alone cannot.
            active = bool(ds.probe_rows(drive_root, "progress", [("chat_id", project["chat_id"])], limit=1))
        if not active and absence_covered is False:  # None: already unknown for this request
            try:
                for stream in ("chat", "progress"):
                    ds.require_imported(drive_root, stream)
                    if ds.range_coverage(drive_root, stream)["gaps"]:
                        raise ds.StoreError("dialogue history has gaps; room activity is unknown")
                absence_covered = True
            except (ds.StoreMigrating, ds.StoreError):
                if chat_id is not None:
                    raise  # the addressed room answers its typed outcome
                # The sidebar keeps every room while the archive imports: an unproved room reads as
                # not yet active (as before the import found its rows), never as a missing project.
                absence_covered = None
        out.append({
            "id": project.get("id"),
            "name": project.get("name"),
            "pinned": project.get("pinned") is True,
            "chat_id": project.get("chat_id"),
            "working_dir": project.get("working_dir") or "",
            "cognition": project_cognition(project, cognition),
            "model_choice": project_model_choice(project),
            "provenance": project.get("provenance") or "",
            "last_active_at": project.get("last_active_at") or "",
            **({"closed": dict(project["closed"])} if "closed" in project else {}),
            "lifecycle": project.get("lifecycle") or PROJECT_ACTIVE,
            "routing_generation": int(project.get("routing_generation") or 0),
            "visible_revision": int(project.get("visible_revision") or 0),
            "delete_error": project.get("delete_error") or "",
            "has_thread_activity": active,
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
    "list_projects",
    "list_reserved_projects",
    "list_sidebar_projects",
    "origin_claim_lock",
    "origin_key",
    "project_binding_for_task",
    "project_chat_for_task",
    "project_thread_note_for_task",
    "project_chat_for_task_tree",
    "RoomResolver",
    "project_id_for_origin",
    "project_task_bindings",
    "task_presentation_snapshot",
    "task_presentation_name",
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

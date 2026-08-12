"""Project threads — storage rows, canonical projection, chat-id reservation.

Lifted verbatim out of :mod:`ouroboros.projects_registry` (no behaviour change)
so the registry stays under the module-size gate while the thread lifecycle
keeps growing. The seam is real rather than convenient: thread #0 is never
STORED — it is projected at read time from the project row itself — so every
member here reads or appends the ADDITIVE ``threads: []`` list and nothing in
the registry's project lifecycle depends on thread state.

The registry keeps its own locking/IO primitives (``_file_write_lock``,
``_load``, ``_save``, ``_registry_path``, ``_validated_name``) and imports this
module at module level to re-export the public thread surface, so the borrow
direction here is deliberately deferred: every registry primitive is imported
INSIDE the function that needs it. That keeps the import graph one-way
(registry -> threads) and, because a function-local ``from ... import`` resolves
the attribute at call time, it keeps the registry module the single place where
those primitives can be patched or replaced.
"""

from __future__ import annotations

import logging
import pathlib
from typing import Any, Dict, List, Optional

from ouroboros.contracts.chat_id_policy import MAIN_THREAD_ID, project_chat_id, thread_chat_id
from ouroboros.project_facts import sanitize_project_id
from ouroboros.utils import utc_now_iso

log = logging.getLogger(__name__)

# Thread lifecycle (D4/X10). The project pattern, one level down:
#   active      the ordinary state
#   archived    hidden, RESTORABLE, and everything — history, worktree, bindings
#               — left exactly as it was. An archived thread whose task is still
#               running stays VISIBLE until that task is terminal (the owner-
#               locked reading of X10): hiding live output would leave work
#               emitting into a room nobody can see.
#   deleting    fenced against routing, its live tasks cancelling; still visible
#               while it quiesces, exactly as a deleting PROJECT is
#   tombstoned  terminal. The id and chat id are never reused, and the journal
#               rows physically remain — stated honestly rather than claimed
#               erased, because nothing here rewrites the shared journal.
THREAD_ACTIVE = "active"
THREAD_ARCHIVED = "archived"
THREAD_DELETING = "deleting"
THREAD_TOMBSTONED = "tombstoned"
THREAD_LIFECYCLES = frozenset({THREAD_ACTIVE, THREAD_ARCHIVED, THREAD_DELETING, THREAD_TOMBSTONED})
#: Lifecycles a thread may be seen in at all. A tombstone is never shown; whether
#: an ARCHIVED thread is shown depends on whether it is still live (X10).
_SIDEBAR_LIFECYCLES = frozenset({THREAD_ACTIVE, THREAD_DELETING})

# Bound the retry walk when a minted thread chat id is already reserved. Each
# step is a fresh deterministic pre-image, so exhausting this many is a
# registry-wide alarm, not a routine outcome.
_THREAD_ID_MINT_ATTEMPTS = 64
# Registry VERSIONS (path, mtime_ns, size) whose duplicate-chat-id scan already
# ran — the scan is an alarm, not a per-read log flood, but keying it on the
# file version means a collision hand-edited in later is still reported. Bounded
# so a long-lived writer process cannot accumulate one entry per write.
_DUPLICATE_CHAT_ID_REPORTED: set = set()
_DUPLICATE_MEMO_MAX = 64


def _normalize_thread_rows(value: Any) -> List[Dict[str, Any]]:
    """Normalize the ADDITIVE extra-thread list of a project row (read-only).

    A legacy row has no ``threads`` key at all and normalizes to ``[]`` — i.e.
    exactly one (projected) thread. Rows are dropped rather than repaired when
    they carry no usable id/chat_id, and thread id ``0`` is never accepted from
    storage: thread #0 is synthesized from the project itself, so a stored
    duplicate would be a second, silently disagreeing truth.

    Like ``_normalize_project_row``, each row is built from ``dict(raw)`` and
    only the normalized keys are OVERWRITTEN. Rebuilding a fresh dict of known
    keys silently DELETED every additive field a later phase adds, and
    normalization runs on every ``_load``, so the loss would look like data that
    was never written.
    """
    out: List[Dict[str, Any]] = []
    seen_ids: set = set()
    if not isinstance(value, list):
        return out
    for raw in value:
        if not isinstance(raw, dict):
            continue
        try:
            thread_id = int(raw.get("id"))
            chat_id = int(raw.get("chat_id"))
        except (TypeError, ValueError):
            continue
        if thread_id <= MAIN_THREAD_ID or not chat_id or thread_id in seen_ids:
            continue
        seen_ids.add(thread_id)
        lifecycle = str(raw.get("lifecycle") or THREAD_ACTIVE).strip().lower()
        row: Dict[str, Any] = dict(raw)
        row["id"] = thread_id
        row["chat_id"] = chat_id
        row["name"] = str(raw.get("name") or "").strip() or f"Thread {thread_id}"
        row["created_at"] = str(raw.get("created_at") or "")
        # A legacy row has no lifecycle at all and reads as active — the only
        # state that existed when it was written.
        row["lifecycle"] = lifecycle if lifecycle in THREAD_LIFECYCLES else THREAD_ACTIVE
        row["archived_at"] = str(raw.get("archived_at") or "")
        row["delete_error"] = str(raw.get("delete_error") or "")
        try:
            row["visible_revision"] = max(0, int(raw.get("visible_revision") or 0))
        except (TypeError, ValueError):
            row["visible_revision"] = 0
        # Fork cursor (A3): a pointer into the PARENT's rows, never a copy. A
        # HALF-written cursor is not a cursor — drop both keys so an ancestry
        # walk can never inherit a bound without a parent (or the reverse).
        try:
            fork_of = int(raw.get("fork_of_chat_id") or 0)
        except (TypeError, ValueError):
            fork_of = 0
        fork_before = str(raw.get("fork_before_ts") or "")
        if fork_of and fork_before:
            row["fork_of_chat_id"] = fork_of
            row["fork_before_ts"] = fork_before
        else:
            row.pop("fork_of_chat_id", None)
            row.pop("fork_before_ts", None)
        out.append(row)
    return sorted(out, key=lambda r: int(r["id"]))


def project_threads(project: Dict[str, Any]) -> List[Dict[str, Any]]:
    """CANONICAL thread projection of a project row — thread #0 first.

    Thread #0 is SYNTHESIZED from the project's own ``chat_id``/``name``/
    ``created_at`` (X7): nothing on disk is rewritten, and the top-level
    ``chat_id`` remains its compatibility alias. Every consumer that wants "the
    threads of this project" must read THIS, never the raw ``threads`` list, or
    it will silently lose the project's main thread.

    Its revision comes from the project's OWN ``thread0_visible_revision``, not
    from ``visible_revision`` — that one is the project-wide AGGREGATE, so
    projecting it here made activity in ANY thread read as unread activity in
    thread #0. A row that predates the split falls back to the aggregate, which
    is the same number while a project has one thread.
    """
    if not isinstance(project, dict):
        return []
    pid = str(project.get("id") or "")
    try:
        chat_id = int(project.get("chat_id") or 0)
    except (TypeError, ValueError):
        chat_id = 0
    try:
        own_revision = max(0, int(
            project.get(
                "thread0_visible_revision", project.get("visible_revision")
            ) or 0
        ))
    except (TypeError, ValueError):
        own_revision = 0
    zero = {
        "id": MAIN_THREAD_ID,
        "chat_id": chat_id or project_chat_id(pid),
        "name": str(project.get("name") or pid),
        "created_at": str(project.get("created_at") or ""),
        "visible_revision": own_revision,
        # Thread #0 IS the project, so it has no lifecycle of its own to hold: it
        # mirrors the project's, and archiving or deleting it is refused in
        # favour of the project's own operations. A synthesized state that could
        # disagree with the project row would be a second truth about one thing.
        "lifecycle": _thread_zero_lifecycle(project),
        "archived_at": "",
        "delete_error": str(project.get("delete_error") or ""),
    }
    return [zero, *_normalize_thread_rows(project.get("threads"))]


def _thread_zero_lifecycle(project: Dict[str, Any]) -> str:
    """The project's own lifecycle, spoken in the thread vocabulary."""
    from ouroboros.projects_registry import (
        PROJECT_ACTIVE,
        PROJECT_DELETING,
        PROJECT_TOMBSTONED,
    )

    return {
        PROJECT_ACTIVE: THREAD_ACTIVE,
        PROJECT_DELETING: THREAD_DELETING,
        PROJECT_TOMBSTONED: THREAD_TOMBSTONED,
    }.get(str(project.get("lifecycle") or PROJECT_ACTIVE), THREAD_ACTIVE)


def thread_is_visible(
    thread: Dict[str, Any], live_chat_ids: Any = None, *, include_archived: bool = False,
) -> bool:
    """Should this thread be SHOWN? (D4 + X10's owner-locked reading.)

    ``active`` and ``deleting`` are always visible — a deleting thread stays on
    screen while it quiesces, exactly as a deleting project does. ``archived`` is
    hidden UNLESS a task is still live in it: archiving is a filing gesture, not a
    kill switch, and hiding a room that is still emitting output would leave the
    owner watching nothing while work continues. ``tombstoned`` is never shown.

    ``live_chat_ids`` is supplied by the CALLER (the gateway reads the queue).
    This module stays pure: a registry projection that reached into the supervisor
    queue would make every read of the sidebar depend on the queue lock.

    ``include_archived`` is the ASKING-FOR-THEM case, and it exists because the
    default made archive a ONE-WAY trip. Archived threads were filtered out of
    the only projection that lists threads, so nothing the owner could see ever
    carried an archived thread — which made ``POST …/restore`` and the ``restore``
    row in the thread menu unreachable BY CONSTRUCTION. Restoring something
    requires a surface that can show it first. ``tombstoned`` is still never
    shown: that one really is gone.
    """
    lifecycle = str(thread.get("lifecycle") or THREAD_ACTIVE)
    if lifecycle in _SIDEBAR_LIFECYCLES:
        return True
    if lifecycle != THREAD_ARCHIVED:
        return False
    if include_archived:
        return True
    try:
        return int(thread.get("chat_id") or 0) in set(live_chat_ids or ())
    except (TypeError, ValueError):
        return False


def _row_chat_ids(project: Dict[str, Any]) -> List[int]:
    """Every chat id a single project row reserves (thread #0 included)."""
    return [int(thread["chat_id"]) for thread in project_threads(project)]


def _chat_id_owners(projects: List[Dict[str, Any]]) -> Dict[int, List[tuple]]:
    """chat_id -> [(project_id, thread_id), ...] across EVERY lifecycle state.

    Tombstoned rows keep reserving their ids on purpose: a reused chat id would
    silently merge a dead project's history into a live one.
    """
    owners: Dict[int, List[tuple]] = {}
    for project in projects:
        pid = str(project.get("id") or "")
        for thread in project_threads(project):
            owners.setdefault(int(thread["chat_id"]), []).append((pid, int(thread["id"])))
    return owners


def duplicate_chat_ids(drive_root: Any) -> Dict[int, List[tuple]]:
    """Pre-existing chat-id collisions in the registry (X1 load-time detection).

    Returns only ids claimed by more than one (project, thread) pair. A healthy
    registry returns ``{}``; anything else means two conversations would share
    one history stream and must be surfaced, never silently tolerated.
    """
    from ouroboros.projects_registry import _LOCK, _load

    with _LOCK:
        projects = _load(drive_root)["projects"]
    return {cid: owners for cid, owners in _chat_id_owners(projects).items() if len(owners) > 1}


def _report_duplicate_chat_ids(drive_root: Any, projects: List[Dict[str, Any]]) -> None:
    """Loudly report duplicates once per registry VERSION per drive root.

    Called from ``_load`` so a corrupt registry cannot stay quiet; deliberately
    non-raising, because refusing to load the registry would take the whole
    server down over data that is still individually readable. The memo is
    keyed on the file's (mtime, size) rather than the root alone, so a
    hand-edited collision introduced AFTER the first load is still reported
    instead of hiding behind a once-per-process flag.
    """
    from ouroboros.projects_registry import _registry_path

    path = _registry_path(drive_root)
    try:
        stat = path.stat()
        key = (str(path), stat.st_mtime_ns, stat.st_size)
    except OSError:
        key = (str(path), 0, 0)
    if key in _DUPLICATE_CHAT_ID_REPORTED:
        return
    clashes = {cid: owners for cid, owners in _chat_id_owners(projects).items() if len(owners) > 1}
    if len(_DUPLICATE_CHAT_ID_REPORTED) >= _DUPLICATE_MEMO_MAX:
        _DUPLICATE_CHAT_ID_REPORTED.clear()  # bounded: at worst one extra scan
    _DUPLICATE_CHAT_ID_REPORTED.add(key)
    if not clashes:
        return
    log.error(
        "Project registry chat-id COLLISION: %s — these conversations share one "
        "history stream; rename/recreate one of them",
        {cid: owners for cid, owners in sorted(clashes.items())},
    )
    try:
        from ouroboros.utils import append_jsonl

        append_jsonl(
            pathlib.Path(drive_root) / "logs" / "events.jsonl",
            {
                "ts": utc_now_iso(),
                "type": "project_chat_id_collision_detected",
                "collisions": {str(cid): owners for cid, owners in sorted(clashes.items())},
            },
        )
    except Exception:
        log.debug("Failed to record chat-id collision event", exc_info=True)


def get_thread(drive_root: Any, project_id: str, thread_id: Any) -> Optional[Dict[str, Any]]:
    """One thread of a project by id (thread #0 included), else ``None``."""
    from ouroboros.projects_registry import get_reserved_project

    project = get_reserved_project(drive_root, project_id)
    if project is None:
        return None
    try:
        want = int(thread_id)
    except (TypeError, ValueError):
        return None
    for thread in project_threads(project):
        if int(thread["id"]) == want:
            return dict(thread)
    return None


def _mint_thread(data: Dict[str, Any], entry: Dict[str, Any], existing: List[Dict[str, Any]]) -> tuple:
    """Pick the next free ``(thread_id, chat_id)`` pair for a project row.

    Thread ids are opaque integers, so a chat-id collision is resolved by simply
    walking to the next id (X1's "retry thread ids on collision") — no allocator
    state, no widened hash. Raises when the walk is exhausted, which is a
    registry-wide alarm rather than a routine outcome.

    The high-water mark is persisted as ``thread_seq`` rather than derived from
    the live rows alone, so a thread id is NEVER reused once handed out. That
    matters the moment threads become removable: a reused id would mint a chat
    id a dead thread's history rows still carry, silently merging two
    conversations. A legacy row has no ``thread_seq`` and falls back to the
    live maximum, which is correct while nothing has been removed yet.
    """
    pid = str(entry.get("id") or "")
    reserved = _chat_id_owners(data["projects"])
    try:
        high_water = max(0, int(entry.get("thread_seq") or 0))
    except (TypeError, ValueError):
        high_water = 0
    next_id = max(
        high_water, max((int(row["id"]) for row in existing), default=MAIN_THREAD_ID)
    ) + 1
    for candidate in range(next_id, next_id + _THREAD_ID_MINT_ATTEMPTS):
        chat_id = thread_chat_id(pid, candidate)
        if chat_id not in reserved:
            return candidate, chat_id
    raise ValueError(
        f"could not mint a free thread chat id for project {pid!r} after "
        f"{_THREAD_ID_MINT_ATTEMPTS} attempts — the registry has a chat-id collision storm"
    )


def _active_project_row(data: Dict[str, Any], pid: str) -> Dict[str, Any]:
    """The project row a thread mutation may write to, or a TYPED refusal.

    A project that is deleting or tombstoned refusing thread changes is a
    PRECONDITION the owner can read and act on — the project is on its way out —
    so it answers like every other lifecycle refusal in this module and reaches
    the routes as a 409. A bare ``ValueError`` here reached ``json_exception``
    instead and became a 500: the same fact, rendered as a crash, with no reason
    a UI could branch on (T3R-17).
    """
    from ouroboros.projects_registry import PROJECT_ACTIVE

    for entry in data["projects"]:
        if entry.get("id") == pid:
            lifecycle = str(entry.get("lifecycle") or PROJECT_ACTIVE)
            if lifecycle != PROJECT_ACTIVE:
                raise ThreadLifecycleError(
                    "project_not_active",
                    f"This project is {lifecycle}, so its threads cannot be changed.",
                )
            return entry
    raise ThreadLifecycleError("unknown_project", f"unknown project: {pid!r}")


def create_thread(
    drive_root: Any,
    project_id: str,
    *,
    name: str = "",
    fork_of_chat_id: int = 0,
    fork_before_ts: str = "",
) -> Dict[str, Any]:
    """Append a NEW thread to a project and return its canonical row.

    A thread is an empty chat sharing the project's working folder (A2). The
    fork variant stores only a CURSOR ``{fork_of_chat_id, fork_before_ts}``
    (A3) — no history row is copied, so the parent keeps one row identity, one
    consolidation and one rotation cost. Prefer :func:`fork_thread` for forks;
    this is the primitive both paths share.
    """
    from ouroboros.projects_registry import (
        _file_write_lock,
        _load,
        _registry_path,
        _save,
        _validated_name,
    )

    pid = sanitize_project_id(project_id)
    if not pid:
        raise ValueError(f"unusable project id: {project_id!r}")
    title = _validated_name(name, "New thread")
    with _file_write_lock(_registry_path(drive_root)):
        data = _load(drive_root)
        entry = _active_project_row(data, pid)
        threads = _normalize_thread_rows(entry.get("threads"))
        thread_id, chat_id = _mint_thread(data, entry, threads)
        row: Dict[str, Any] = {
            "id": thread_id,
            "chat_id": chat_id,
            "name": title,
            "created_at": utc_now_iso(),
            "visible_revision": 0,
        }
        if fork_of_chat_id and fork_before_ts:
            row["fork_of_chat_id"] = int(fork_of_chat_id)
            row["fork_before_ts"] = str(fork_before_ts)
        entry["threads"] = [*threads, row]
        # Monotonic high-water mark: a thread id is never handed out twice.
        entry["thread_seq"] = thread_id
        _save(drive_root, data)
        log.info("Project thread created: %s#%s (chat_id=%s)", pid, thread_id, chat_id)
        return dict(row)


def rename_thread(drive_root: Any, project_id: str, thread_id: Any, name: str) -> Optional[Dict[str, Any]]:
    """Rename a thread. Thread #0 IS the project, so renaming it renames the
    project row itself — the projection would otherwise show a name the sidebar
    never persists."""
    from ouroboros.projects_registry import (
        _file_write_lock,
        _load,
        _registry_path,
        _save,
        _validated_name,
        update_project,
    )

    pid = sanitize_project_id(project_id)
    if not pid:
        return None
    try:
        want = int(thread_id)
    except (TypeError, ValueError):
        return None
    title = _validated_name(name)
    if not title:
        raise ValueError("thread name is required")
    if want == MAIN_THREAD_ID:
        updated = update_project(drive_root, pid, name=title)
        return project_threads(updated)[0] if updated else None
    with _file_write_lock(_registry_path(drive_root)):
        data = _load(drive_root)
        entry = _active_project_row(data, pid)
        threads = _normalize_thread_rows(entry.get("threads"))
        for row in threads:
            if int(row["id"]) == want:
                row["name"] = title
                entry["threads"] = threads
                _save(drive_root, data)
                return dict(row)
    return None


class ThreadLifecycleError(ValueError):
    """A lifecycle transition that must be REFUSED, with a typed ``reason``."""

    def __init__(self, reason: str, message: str) -> None:
        self.reason = reason
        super().__init__(message)


def _set_thread_lifecycle(
    drive_root: Any,
    project_id: str,
    thread_id: Any,
    *,
    to: str,
    allowed_from: frozenset,
    stamp_archived: bool = False,
    clear_archived: bool = False,
    delete_error: Any = None,
) -> Optional[Dict[str, Any]]:
    """The ONE writer of a thread's lifecycle field.

    Every transition goes through here so the guards — thread #0 is the project,
    a tombstone is terminal, an unexpected source state is refused rather than
    overwritten — are written once and cannot drift between archive, restore,
    fence and tombstone.
    """
    from ouroboros.projects_registry import _file_write_lock, _load, _registry_path, _save

    pid = sanitize_project_id(project_id)
    if not pid:
        return None
    try:
        want = int(thread_id)
    except (TypeError, ValueError):
        return None
    if want == MAIN_THREAD_ID:
        raise ThreadLifecycleError(
            "thread_zero_is_the_project",
            "This thread IS the project. Archive or delete the project itself.",
        )
    with _file_write_lock(_registry_path(drive_root)):
        data = _load(drive_root)
        entry = _active_project_row(data, pid)
        threads = _normalize_thread_rows(entry.get("threads"))
        for row in threads:
            if int(row["id"]) != want:
                continue
            current = str(row.get("lifecycle") or THREAD_ACTIVE)
            if current == to:
                return dict(row)  # idempotent: the caller may have retried
            if current not in allowed_from:
                raise ThreadLifecycleError(
                    "lifecycle_conflict",
                    f"this thread is {current}; it cannot become {to}",
                )
            row["lifecycle"] = to
            if stamp_archived:
                row["archived_at"] = utc_now_iso()
            if clear_archived:
                row["archived_at"] = ""
            if delete_error is not None:
                row["delete_error"] = str(delete_error)
            entry["threads"] = threads
            _save(drive_root, data)
            log.info("Project thread %s#%s -> %s", pid, want, to)
            return dict(row)
    return None


def archive_thread(drive_root: Any, project_id: str, thread_id: Any) -> Optional[Dict[str, Any]]:
    """Hide a thread. NOTHING is removed and everything is restorable (D4).

    History rows, task bindings, the fork cursors of its children and its git
    worktree all stay exactly as they are — archiving is a filing gesture, so the
    only thing it changes is whether the thread appears in the list. An archived
    thread with a task still running stays VISIBLE until that task is terminal
    (:func:`thread_is_visible`), because hiding a room that is still emitting
    output is how an owner ends up watching nothing while work continues.
    """
    return _set_thread_lifecycle(
        drive_root, project_id, thread_id,
        to=THREAD_ARCHIVED,
        allowed_from=frozenset({THREAD_ACTIVE}),
        stamp_archived=True,
    )


def restore_thread(drive_root: Any, project_id: str, thread_id: Any) -> Optional[Dict[str, Any]]:
    """Un-archive. The inverse of :func:`archive_thread`, and nothing more."""
    return _set_thread_lifecycle(
        drive_root, project_id, thread_id,
        to=THREAD_ACTIVE,
        allowed_from=frozenset({THREAD_ARCHIVED}),
        clear_archived=True,
    )


def begin_thread_deletion(drive_root: Any, project_id: str, thread_id: Any) -> Optional[Dict[str, Any]]:
    """FENCE a thread for deletion — the first of X10's three steps.

    Marking ``deleting`` is what closes routing into the thread's chat before any
    cancellation starts (``resolve_chat_binding`` reports it, and the routing
    classifier refuses a non-active thread). Doing it in the other order would let
    a message land in a room that is on its way out.

    An ARCHIVED thread may be deleted directly — the owner filing something away
    and then discarding it is one flow, not two.
    """
    return _set_thread_lifecycle(
        drive_root, project_id, thread_id,
        to=THREAD_DELETING,
        allowed_from=frozenset({THREAD_ACTIVE, THREAD_ARCHIVED}),
        delete_error="",
    )


def complete_thread_deletion(drive_root: Any, project_id: str, thread_id: Any) -> Optional[Dict[str, Any]]:
    """TOMBSTONE a fenced thread once its tasks have quiesced (X10's third step).

    The row STAYS. Its id and chat id keep their reservation forever — with 28-bit
    chat ids a reused one would silently merge a dead thread's history into a live
    conversation — and the journal rows physically remain, which is stated rather
    than dressed up as erasure: the journal is shared by every chat and nothing
    here rewrites it. What a tombstone buys is that the thread is gone from every
    surface and can never come back.
    """
    return _set_thread_lifecycle(
        drive_root, project_id, thread_id,
        to=THREAD_TOMBSTONED,
        allowed_from=frozenset({THREAD_DELETING}),
        delete_error="",
    )


def fail_thread_deletion(
    drive_root: Any, project_id: str, thread_id: Any, error: str,
) -> Optional[Dict[str, Any]]:
    """Record WHY a deletion could not quiesce, leaving the thread fenced.

    Deliberately not a rollback to ``active``: routing stays closed, because the
    owner asked for this thread to go and a silent un-fence would put messages
    back into a room they had written off. The error is stored so the surface can
    say what is stuck instead of showing a thread that never finishes vanishing.
    """
    from ouroboros.projects_registry import _file_write_lock, _load, _registry_path, _save

    pid = sanitize_project_id(project_id)
    try:
        want = int(thread_id)
    except (TypeError, ValueError):
        return None
    with _file_write_lock(_registry_path(drive_root)):
        data = _load(drive_root)
        entry = _active_project_row(data, pid)
        threads = _normalize_thread_rows(entry.get("threads"))
        for row in threads:
            if int(row["id"]) == want and str(row.get("lifecycle")) == THREAD_DELETING:
                row["delete_error"] = str(error or "")[:500]
                entry["threads"] = threads
                _save(drive_root, data)
                return dict(row)
    return None


def fork_thread(drive_root: Any, project_id: str, thread_id: Any) -> Dict[str, Any]:
    """Fork a thread: a new thread carrying a CURSOR into the source's rows.

    The source is untouched and keeps every row (A3). The cursor reads the
    parent's rows REGARDLESS of the parent later being archived or deleted
    (A3a), so a fork can never be orphaned. Auto-name is the plain English
    ``Copy of …`` with NO model call (D2).
    """
    from ouroboros.projects_registry import THREAD_NAME_MAX

    source = get_thread(drive_root, project_id, thread_id)
    if source is None:
        raise ValueError(f"unknown thread {thread_id!r} in project {project_id!r}")
    label = str(source.get("name") or "").strip()
    auto = f"Copy of {label}" if label else "Copy of thread"
    return create_thread(
        drive_root,
        project_id,
        name=auto[:THREAD_NAME_MAX],
        fork_of_chat_id=int(source["chat_id"]),
        # The fork moment. History treats it INCLUSIVELY (``ts <= cutoff``):
        # a parent row stamped at exactly this instant existed before the fork.
        fork_before_ts=utc_now_iso(),
    )


def project_thread_note_for_task(task: Any) -> str:
    """One-line pointer to the Project thread when a task is project-bound.

    The raw final answer of a bound task lives in the PROJECT room while the
    initiating (Main) chat receives only the task summary — twice in one night
    the owner read that silence as a hung agent. The pointer names where the
    full result lives (v6.70.0); an unbound task gets no extra text."""
    try:
        import pathlib as _pathlib

        from ouroboros.config import DATA_DIR
        from ouroboros.projects_registry import list_projects, project_chat_for_task_tree

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


__all__ = [
    "THREAD_ACTIVE",
    "THREAD_ARCHIVED",
    "THREAD_DELETING",
    "THREAD_LIFECYCLES",
    "THREAD_TOMBSTONED",
    "ThreadLifecycleError",
    "archive_thread",
    "begin_thread_deletion",
    "complete_thread_deletion",
    "create_thread",
    "fail_thread_deletion",
    "duplicate_chat_ids",
    "fork_thread",
    "get_thread",
    "project_thread_note_for_task",
    "project_threads",
    "rename_thread",
    "restore_thread",
    "thread_is_visible",
]

"""The owner's direct "keep this": one saved fact on the exact named task's own document.

``owner_keep: {at, scope: "task", source, note?}`` holds that task's materials only: never a descendant
or a line, and several tasks need several facts. ``source`` is what the host knows of the request:
the authenticated gateway route stamps its own request, the ``set_task_retention`` tool the owner's
message the turn holds; a call without such a source writes nothing, keep or release alike.

Every retirer of a task's material reads the fact inside ``gate`` around its effect; the writer saves it
under the same per-task custody lock, so an answer of "kept" never follows a move or removal already
accepted. A child scratch handoff stays owned until its record accepts deletion under this lock;
Keep refreshes relocated handoffs here. Root and seat handoffs keep their earlier meaning. The fact only withholds deletion: it never stops a running task, its review captures or
lossless compression (``subagent_worktrees._task_proven_over`` does not read it). A queue hold and a
producer's retention marker mean other things and are not read here. An engine reviewer thread goes only
through a release operation bound on this document inside ``gate`` (``review_thread_retention``); every thread
the task's plan seats named or started (``seat_threads``) is its material before any pass reaches it, and a keep saved
first prevents that binding. A bound operation is only an intent in flight, not the engine's acceptance:
a keep saved after it is saved for everything still owned and as the intention to stop that release, which the
next pass abandons unless the engine already accepted it (a retire already delivered can still win). Such a
thread is answered ``release_unconfirmed``, never kept, and only an engine cleanup result already observed is
called removed: no answer polls the engine.

Lock order: an object's own lock, then the custody lock, always with a bounded wait. Nothing that may
wait for the supervisor's queue lock runs inside ``gate``: the liveness probe
(``supervisor.queue_transitions.task_settlement_liveness``) takes that lock, and drive settlement holds it
while it waits for custody, so a retirer asks the probe before entering ``gate`` and re-checks only
recorded facts inside. The writer takes the custody lock alone, reads its inventory before it and holds
no store transaction across a wait.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import pathlib
from typing import Any, Dict, Iterator, List, Literal, Optional, Tuple

try:  # Python 3.11+
    from typing import TypedDict  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - CI supports Python 3.10.
    from typing_extensions import TypedDict  # type: ignore[assignment]

log = logging.getLogger(__name__)

OWNER_KEEP = "owner_keep"
SCOPE_TASK = "task"
_NOTE_MAX = 500


class OwnerKeepRequest(TypedDict, total=False):
    """``POST /api/tasks/{task_id}/owner-keep``: keep or release the materials of exactly this task."""

    keep: bool
    note: str


class OwnerKeepResponse(TypedDict):
    task_id: str
    keep: bool  # the fact as it now stands
    status: Literal["kept", "released", "partly_retired", "already_retired", "release_unconfirmed", "inflight",
                    "refused"]
    reason: str  # why inflight or refused; else empty
    retained: List[str]  # material found on disk now: paths, ``pins:<snapshot>`` for result pins,
    # ``saved_states:<folder>`` while that folder holds a saved state for exact return, and
    # ``review_thread:<id>`` for an engine reviewer thread no recorded release has let go
    retired: List[str]  # gone, accepted for deletion, or managed import ID reuse released; not proof of deleted bytes
    unknown: List[str]  # material classes that could not be read: nothing is claimed about them
    releases: Dict[str, str]  # each ``review_thread:<id>`` -> its release as last observed, never polled:
    # ``release_unconfirmed`` (bound; neither kept nor gone), ``accepted_pending_removal``, ``removed`` (the engine's
    # cleanup result said so), ``purged`` (by the engine's own route), ``not_found`` (the store it asked does not hold
    # it: no purge, in neither list, ``unknown`` names ``review_threads``), or, retained, ``unrecorded`` (a plan seat
    # named or started it, no release is recorded) and ``held``/``refused``/``abandoned``;
    # ``managed_import:<key>`` -> held | release_unconfirmed | future_reuse_released | upload_cancelled |
    # binding_unavailable:<outcome> | release_refused:<code>; none proves deletion of bytes
    # Canonical files -> held | queued (not moved) | moved (not accounted) | staged | deleting | deleted |
    # missing | unknown. Moved/staged/deleting are retired from Keep, not proof of deleted bytes.


def _valid(task_id: Any) -> bool:
    from ouroboros.task_results import validate_task_id

    try:
        validate_task_id(task_id)
        return True
    except ValueError:
        return False


def is_kept(root: Any, task_id: Any) -> bool:
    """Has the owner kept this exact task's materials? An unreadable document counts as kept;
    a name that is no task id, or a task with no document, carries no fact."""
    from ouroboros.task_results import load_task_result

    if not _valid(task_id):
        return False
    try:
        return isinstance((load_task_result(root, str(task_id), strict=True) or {}).get(OWNER_KEEP), dict)
    except Exception:
        return True


@contextlib.contextmanager
def gate(root: Any, task_id: Any, *, timeout_sec: float = 2.0) -> Iterator[str]:
    """Why this task's material may not go now (``owner_keep``, ``busy``), or ``""``, with the
    task's custody lock held around the caller's effect. Bounded, so taking it under an object's own
    lock cannot deadlock; a material with no valid owning task has nothing to read."""
    from ouroboros.install_identity import StoreForeign, require_owned_root
    from ouroboros.task_custody import task_custody_lock

    if not _valid(task_id):
        yield ""
        return
    try:
        require_owned_root(root)
    except StoreForeign:  # another install's store: its keep cannot be read here, so nothing of it goes
        yield "busy"
        return
    with task_custody_lock(root, str(task_id), timeout_sec=timeout_sec) as locked:
        yield "busy" if not locked else OWNER_KEEP if is_kept(root, task_id) else ""


def _json(path: pathlib.Path) -> Dict[str, Any]:
    row = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(row, dict):
        raise ValueError(f"{path} is not an object")
    return row


def _pin_place(copy: Dict[str, Any]) -> str:
    """The inventory's name for the result pins a copy record holds; empty when it holds none."""
    return f"pins:{copy.get('snapshot_id') or copy.get('path')}" if copy.get("baseline_ref") or copy.get("result_ref") else ""


THREAD_PLACE = "review_thread:"
SAVED_STATES_PLACE = "saved_states:"  # a folder's saved states for exact return: there while it still holds one


def _thread_places(row: Dict[str, Any]) -> List[str]:
    """The engine reviewer threads this task's document itself names: each one a reply in its plan record declares
    (``review_thread_retention.review_threads``), named before any pass reaches it, and each one it records a release
    or a store's answer of. ``inventory`` adds what its seats started; this read needs no other store."""
    from ouroboros.review_thread_retention import RELEASES, review_threads

    releases = row.get(RELEASES) if isinstance(row.get(RELEASES), dict) else {}
    threads = {*review_threads(row), *(thread for thread, record in releases.items() if isinstance(record, dict))}
    return [THREAD_PLACE + str(thread) for thread in sorted(threads)]


RELEASE_UNCONFIRMED = "release_unconfirmed"
_REUSE_RELEASED = "future_reuse_released"
_THREAD_GONE = frozenset({"accepted_pending_removal", "removed", "purged", _REUSE_RELEASED, "upload_cancelled"})
_THREAD_ABSENT = "not_found"
_THREADS_CLASS = "review_threads"  # the ``unknown`` class of a thread nothing here can place


def _thread_release(row: Dict[str, Any], place: str) -> str:
    """One engine-thread release as the host last observed it: ``release_unconfirmed`` while bound;
    ``removed`` only when the engine's own cleanup result said so, else ``accepted_pending_removal`` once accepted;
    ``purged`` when the engine reported its own purge; ``unrecorded`` while no release is recorded; otherwise the
    record's state, ``not_found`` included (scoped to its ``store_id``). None of these is read from the engine now."""
    from ouroboros.review_thread_retention import RELEASES

    record = ((row.get(RELEASES) or {}) if isinstance(row.get(RELEASES), dict) else {}).get(place[len(THREAD_PLACE):])
    record = record if isinstance(record, dict) else {}
    engine = record.get("engine") if isinstance(record.get("engine"), dict) else {}
    state = str(record.get("state") or "")
    if state == "bound":
        return RELEASE_UNCONFIRMED
    if state == "accepted":
        return "removed" if engine.get("status") == "removed" else "accepted_pending_removal"
    return "purged" if engine.get("thread_state") == "purged" else state or "unrecorded"


def _thread_present(release: str) -> Optional[bool]:
    """Whether the last observation permits keeping this material or reuse; None when it cannot prove that."""
    return (None if release == _THREAD_ABSENT or release.startswith(("binding_unavailable:", "release_refused:"))
            else release not in _THREAD_GONE)


def inventory(root: pathlib.Path, task_id: str, row: Dict[str, Any]) -> Tuple[List[str], List[str], Dict[str, str]]:
    """Everything a gated retirer could remove for this exact task, as ``(places, unknown, pins)``, ``pins``
    naming the copy path whose record each ``pins:`` place stands on, and the ownership record each session
    scratch stands on. One listing per current folder (copy records, the copies' root, review seats and
    checkouts), the task's canonical artifact tree and one keyed custody read per stopped seat's operation; a listing, marker or
    start that cannot be read names its class in ``unknown``."""
    from ouroboros.copy_environments import ENV_SUFFIX, scratch_owner
    from ouroboros.headless import HEADLESS_TASKS_DIR, TASK_DRIVES_DIR
    from ouroboros.private_copy_read import _copy_owner, read_records
    from ouroboros.record_compression import WIRE_JOURNAL
    from ouroboros.review_thread_retention import seat_threads
    from ouroboros.subagent_worktrees import _record_file, _resolve_root
    from ouroboros.task_custody import child_drive_candidates
    from ouroboros.unindexed import COPIES
    from ouroboros.workspace_preflight import SESSION_SCRATCH_DIRNAME

    constraint = row.get("task_constraint") if isinstance(row.get("task_constraint"), dict) else {}
    places = [str(path) for path in child_drive_candidates(row)] + [str(constraint.get("write_root") or "")]
    places += [str(path) for path in (root / TASK_DRIVES_DIR / task_id, root / HEADLESS_TASKS_DIR / task_id,
                                      root / "task_trees" / task_id,
                                      root / "task_results" / "artifacts" / task_id / WIRE_JOURNAL) if os.path.lexists(path)]
    unknown: List[str] = []
    pins: Dict[str, str] = {}
    from ouroboros.saved_state_retirement import worked_copies_in, worked_copy

    try:  # the folders whose saved states for exact return ``saved_state_retirement.prune_worked_copies`` would retire
        places += sorted({SAVED_STATES_PLACE + str(path.parent)
                          for path, _kind, _info in worked_copies_in(root / "task_results" / "artifacts" / task_id)})
    except OSError:
        unknown.append("saved_states")
    from ouroboros.artifacts import iter_artifact_tree
    from ouroboros.payload_release_state import store_of

    canonical = store_of(root, task_id)
    try:
        try:
            canonical.lstat()
        except FileNotFoundError:
            canonical = None
        if canonical is not None:
            if canonical.resolve(strict=False) != canonical:
                raise OSError("canonical artifact store reached through a link")
            handles = canonical / "source_handles"  # a saved state is named by its folder, never per file
            for path in iter_artifact_tree(canonical):
                if path.is_file() and not path.is_symlink():  # one that arrived after the first walk names its folder
                    saved = path.parent.parent == handles and worked_copy(path.parent.name, path.name)
                    places.append(SAVED_STATES_PLACE + str(path.parent) if saved else str(path))
    except OSError:
        unknown.append("canonical_payloads")
    read = read_records(root)
    if read.unlistable or read.unreadable or read.import_pending:
        unknown.append("private_copies")
    for copy in read.rows:
        if str(copy.get("task_id") or "") != task_id:
            continue
        places.append(str(copy.get("path") or ""))
        if copy.get("path") and os.path.lexists(str(copy["path"]) + ".env"):  # its process environment, if any
            places.append(str(copy["path"]) + ".env")
        if _pin_place(copy):
            places.append(_pin_place(copy))
            pins[_pin_place(copy)] = str(copy.get("path") or "")
    recorded = {str(pathlib.Path(str(copy["path"])).resolve(strict=False)) for copy in read.rows if copy.get("path")}
    try:  # the owner Storage's saved groups gate a copies-root folder on: an environment's declared task, else,
        # as for a folder no record names, the task its name carries
        for folder in (_resolve_root(), _resolve_root() / COPIES):
            for entry in sorted(folder.iterdir()) if folder.is_dir() and not folder.is_symlink() else ():
                if entry.name.endswith(ENV_SUFFIX):
                    named = str((scratch_owner(entry) or {}).get("task_id") or "") or _copy_owner(entry.name[:-len(ENV_SUFFIX)])
                elif entry.name.startswith(".") or entry.name == COPIES or entry.is_symlink() or not entry.is_dir() \
                        or str(entry.resolve(strict=False)) in recorded or read.unlistable or read.import_pending \
                        or _record_file(entry, root).name in read.unreadable:
                    continue
                else:
                    named = _copy_owner(entry.name)
                if named == task_id:
                    places.append(str(entry))
        drive = root / TASK_DRIVES_DIR / task_id / SESSION_SCRATCH_DIRNAME
        for owner in (sorted(drive.iterdir()) if drive.is_dir() else ()):
            with contextlib.suppress(FileNotFoundError):
                scratch = str(_json(owner / "scratch.json").get("path") or "")
                places.append(scratch)
                if scratch:  # the task's while this record names it: a release moves the record first
                    pins[scratch] = str(owner / "scratch.json")
    except (OSError, ValueError):
        unknown.append("environments")
    _child_scratch_places(root, task_id, places, unknown, pins)
    places += _thread_places(row)  # engine material: its release record, not the disk, says where it stands
    try:  # a seat whose reply named no thread is still known by what it started
        started, complete = seat_threads(root, row)
        places += [THREAD_PLACE + thread for thread in started]
    except Exception:  # an unread store: nothing is claimed about what it would name
        complete = False
    if not complete:
        unknown.append(_THREADS_CLASS)
    reviews = root / "state" / "review_checkouts"
    for kind, folders, marker in (("review_seats", reviews / "seats", "checking.json"),
                                  ("review_checkouts", reviews, "checkout.json")):
        try:
            entries = [entry for entry in (sorted(folders.iterdir()) if folders.is_dir() else ())
                       if entry.is_dir() and not entry.is_symlink() and entry.name not in {"seats", "views"}
                       and not entry.name.startswith(".")]
        except OSError:
            unknown.append(kind)
            continue
        for entry in entries:
            try:
                if str(_json(entry / marker).get("task_id") or "") != task_id:
                    continue
                places.append(str(entry))
                if kind == "review_seats":  # the seat's own scratch, held by its home beside the copy
                    for owner in (sorted((entry / SESSION_SCRATCH_DIRNAME).iterdir())
                                  if (entry / SESSION_SCRATCH_DIRNAME).is_dir() else ()):
                        with contextlib.suppress(FileNotFoundError):
                            scratch = str(_json(owner / "scratch.json").get("path") or "")
                            places.append(scratch)
                            if scratch:
                                pins[scratch] = str(owner / "scratch.json")
            except FileNotFoundError:
                continue  # made before its marker: owned through a recorded round, never this task's by name
            except (OSError, ValueError):
                if kind not in unknown:
                    unknown.append(kind)
    return sorted({place for place in places if place}), unknown, pins


def _child_scratch_places(root: pathlib.Path, task_id: str, places: List[str], unknown: List[str],
                           pins: Dict[str, str]) -> None:
    """Refresh a child's unaccepted handoffs; relocation alone no longer releases its scratch."""
    from ouroboros.session_scratch import child_handoff_records

    try:
        for record in child_handoff_records(root, task_id):
            scratch = str(_json(record).get("path") or "")
            if scratch:
                places.append(scratch)
                pins[scratch] = str(record)
    except Exception:  # unread task/store or handoff layout proves no absence
        if "environments" not in unknown:
            unknown.append("environments")


def _present(root: pathlib.Path, task_id: str, place: str, pins: Dict[str, str]) -> Optional[bool]:
    """On disk now; a pin is there while its copy record, one file read under custody, still names it
    for this task. A session scratch stays owned while its record names it: in the drive, or a
    child's handoff before a confirmed release. Root/seat handoff still ends ownership. ``None`` when
    that record cannot be read: nothing is claimed about it."""
    from ouroboros.private_copy_read import record_at

    if place.startswith(SAVED_STATES_PLACE):  # the folder stays with its sources: present while a saved state is left
        from ouroboros.saved_state_retirement import worked_copy
        folder = pathlib.Path(place[len(SAVED_STATES_PLACE):])
        try:
            with os.scandir(folder) as listing:
                return any(worked_copy(folder.name, entry.name) for entry in listing)
        except FileNotFoundError:
            return False
        except OSError:
            return None
    if place not in pins:
        return os.path.lexists(place)
    if not place.startswith("pins:"):
        try:
            from ouroboros.session_scratch import release_finalized

            row = _json(pathlib.Path(pins[place]))
            return not release_finalized(row) and row.get("path") == place and os.path.lexists(place)
        except FileNotFoundError:
            return False
        except (OSError, ValueError):
            return None
    try:
        copy = record_at(pins[place], root) or {}
    except (OSError, ValueError):
        return None
    return str(copy.get("task_id") or "") == task_id and _pin_place(copy) == place


def set_owner_keep(root: Any, task_id: str, *, keep: bool, source: Dict[str, Any], note: str = "",
                   timeout_sec: float = 10.0) -> OwnerKeepResponse:
    """Save or clear the fact under the task's custody lock and answer what its materials are now.
    Inventory is read before locking; under the lock each place and the current document are rechecked.
    Moved material is retired except unaccepted child scratch handoffs, refreshed here and still owned.
    A retirer inside its effect holds that lock: then nothing is saved and the answer is ``inflight``. With a
    complete inventory where everything named is already gone nothing is saved (``already_retired``);
    an unreadable class keeps the fact saved, since material may remain. A bound thread or
    managed-import release saves Keep with ``release_unconfirmed``."""
    from ouroboros.task_custody import task_custody_lock
    from ouroboros.delegate_documents import managed_keep_places
    from ouroboros.payload_release_state import keep_places, store_of
    from ouroboros.task_results import load_task_result
    from ouroboros.task_store import patch_document
    from ouroboros.utils import utc_now_iso

    base = pathlib.Path(root)

    def answer(status: str, stands: bool, *, reason: str = "", retained: Optional[List[str]] = None,
               retired: Optional[List[str]] = None, unknown: Optional[List[str]] = None,
               releases: Optional[Dict[str, str]] = None) -> OwnerKeepResponse:
        return {"task_id": task_id, "keep": stands, "status": status,  # type: ignore[typeddict-item]
                "reason": reason, "retained": retained or [], "retired": retired or [], "unknown": unknown or [],
                "releases": releases or {}}

    if not isinstance(source, dict) or not source.get("basis"):
        return answer("refused", is_kept(base, task_id), reason="source_missing")
    try:
        first = load_task_result(base, task_id, strict=True) or {}
    except Exception:
        return answer("refused", True, reason="task_unreadable")
    if not first:
        return answer("refused", False, reason="task_not_found")
    places, unknown, pins = inventory(base, task_id, first)
    with task_custody_lock(base, task_id, timeout_sec=timeout_sec) as locked:
        if not locked:
            return answer("inflight", is_kept(base, task_id), reason="custody_busy", unknown=unknown)
        row = load_task_result(base, task_id, strict=True) or {}
        stands = isinstance(row.get(OWNER_KEEP), dict)
        _child_scratch_places(base, task_id, places, unknown, pins)
        places = sorted(set(places) | set(_thread_places(row)))  # a thread declared or bound since the first read too
        releases = {place: _thread_release(row, place) for place in places if place.startswith(THREAD_PLACE)}
        try:
            documents = managed_keep_places(base, task_id)
        except Exception:  # unread custody proves no absence, so Keep still saves its fact
            documents = {}
            unknown.append("managed_imports")
        places = sorted(set(places) | set(documents))
        releases.update(documents)
        try:  # fresh canonical facts under the same custody lock as the owner's keep
            payloads = keep_places(base, task_id, row, places)
        except Exception as exc:  # noqa: BLE001 - the rule: an explicit keep is saved whatever this reader does
            # A protective action must not depend on the evidence it protects: an unreadable store or a malformed
            # release record answers ``canonical_payloads`` unknown and the owner's keep is still saved. A release
            # (keep=False) still raises.
            if not keep:
                raise
            log.warning("owner keep of %s: canonical release facts unavailable: %s", task_id, type(exc).__name__)
            store = store_of(base, task_id)  # the listed store files: nothing is claimed about them, the keep is saved
            payloads = {place: ("unknown", None) for place in places if pathlib.Path(place).is_relative_to(store)}
            if "canonical_payloads" not in unknown:
                unknown.append("canonical_payloads")
        places = sorted(set(places) | set(payloads))
        releases.update({place: state for place, (state, _present_now) in payloads.items()})
        releasing = [place for place, state in releases.items() if state == RELEASE_UNCONFIRMED]
        present = {place: payloads[place][1] if place in payloads else _thread_present(releases[place]) if place in releases
                   else _present(base, task_id, place, pins) for place in places if place not in releasing}
        retained = [place for place in present if present[place]]
        retired = [place for place in present if present[place] is False]
        for place in [place for place in present if present[place] is None]:
            kind = ("canonical_payloads" if place in payloads else "managed_imports" if place.startswith("managed_import:") else
                    "saved_states" if place.startswith(SAVED_STATES_PLACE) else
                    "private_copies" if place.startswith("pins:") else _THREADS_CLASS if place in releases
                    else "environments")
            if kind not in unknown:
                unknown.append(kind)
        if keep and retired and not retained and not unknown and not releasing:  # a bound release is no proof of gone
            return answer("already_retired", stands, retired=retired, releases=releases)
        if keep and not stands:
            fact = {"at": utc_now_iso(), "scope": SCOPE_TASK, "source": source,
                    **({"note": note[:_NOTE_MAX]} if note else {})}
            patch_document(base, task_id, lambda current: {**current, OWNER_KEEP: fact})
        elif not keep and stands:
            patch_document(base, task_id, lambda current: {k: v for k, v in current.items() if k != OWNER_KEEP})
    if not keep and stands:  # what the keep held may now go: the retirement step is owed, not left to another wake
        from ouroboros.maintenance_retirement import artifact_changed
        artifact_changed(base, immediate=True)
    status = ("release_unconfirmed" if releasing else "partly_retired" if retired else "kept") if keep else "released"
    return answer(status, keep, retained=retained, retired=retired, unknown=unknown, releases=releases)

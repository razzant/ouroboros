"""The owner's direct "keep this": one saved fact on the exact named task's own document.

``owner_keep: {at, scope: "task", source, note?}`` holds that task's materials only: never a descendant
or a line, and several tasks need several facts. ``source`` is what the host knows of the request:
the authenticated gateway route stamps its own request, the ``set_task_retention`` tool the owner's
message the turn holds; a call without such a source writes nothing, keep or release alike.

Every retirer of a task's material reads the fact inside ``gate`` around its effect; the writer saves it
under the same per-task custody lock, so an answer of "kept" never follows a move or removal already
accepted. The fact only withholds deletion: it never stops a running task, its review captures or
lossless compression (``subagent_worktrees._task_proven_over`` does not read it). A queue hold and a
producer's retention marker mean other things and are not read here.

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
import os
import pathlib
from typing import Any, Dict, Iterator, List, Literal, Optional, Tuple

try:  # Python 3.11+
    from typing import TypedDict  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - CI supports Python 3.10.
    from typing_extensions import TypedDict  # type: ignore[assignment]

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
    status: Literal["kept", "released", "partly_retired", "already_retired", "inflight", "refused"]
    reason: str  # why inflight or refused; else empty
    retained: List[str]  # material found on disk now: paths, and ``pins:<snapshot>`` for result pins
    retired: List[str]  # material a record names that is already gone or accepted for deletion
    unknown: List[str]  # material classes that could not be read: nothing is claimed about them


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


def inventory(root: pathlib.Path, task_id: str, row: Dict[str, Any]) -> Tuple[List[str], List[str], Dict[str, str]]:
    """Everything a gated retirer could remove for this exact task, as ``(places, unknown, pins)``, ``pins``
    naming the copy path whose record each ``pins:`` place stands on. One listing per current folder (copy
    records, the copies' root, review seats and checkouts), no walk of history; a listing or marker that
    cannot be read names its class in ``unknown`` instead."""
    from ouroboros.copy_environments import ENV_SUFFIX, scratch_owner
    from ouroboros.headless import HEADLESS_TASKS_DIR, TASK_DRIVES_DIR
    from ouroboros.private_copy_read import _copy_owner, read_records
    from ouroboros.record_compression import WIRE_JOURNAL
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
    read = read_records(root)
    if read.unlistable or read.unreadable:
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
                        or str(entry.resolve(strict=False)) in recorded or read.unlistable \
                        or _record_file(entry, root).name in read.unreadable:
                    continue
                else:
                    named = _copy_owner(entry.name)
                if named == task_id:
                    places.append(str(entry))
        drive = root / TASK_DRIVES_DIR / task_id / SESSION_SCRATCH_DIRNAME
        for owner in (sorted(drive.iterdir()) if drive.is_dir() else ()):
            with contextlib.suppress(FileNotFoundError):
                places.append(str(_json(owner / "scratch.json").get("path") or ""))
    except (OSError, ValueError):
        unknown.append("environments")
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
                if str(_json(entry / marker).get("task_id") or "") == task_id:
                    places.append(str(entry))
            except FileNotFoundError:
                continue  # made before its marker: owned through a recorded round, never this task's by name
            except (OSError, ValueError):
                if kind not in unknown:
                    unknown.append(kind)
    return sorted({place for place in places if place}), unknown, pins


def _present(root: pathlib.Path, task_id: str, place: str, pins: Dict[str, str]) -> Optional[bool]:
    """On disk now; a pin is there while its copy record, one file read under custody, still names it
    for this task. ``None`` when that record cannot be read: nothing is claimed about the pin."""
    from ouroboros.private_copy_read import record_at

    if place not in pins:
        return os.path.lexists(place)
    try:
        copy = record_at(pins[place], root) or {}
    except (OSError, ValueError):
        return None
    return str(copy.get("task_id") or "") == task_id and _pin_place(copy) == place


def set_owner_keep(root: Any, task_id: str, *, keep: bool, source: Dict[str, Any], note: str = "",
                   timeout_sec: float = 10.0) -> OwnerKeepResponse:
    """Save or clear the fact under the task's custody lock and answer what its materials are now.
    The inventory is read first, without a lock; under the lock each place is checked again and the
    current document decides, so a material moved meanwhile is reported retired, never kept. A retirer
    inside its effect holds that lock: then nothing is saved and the answer is ``inflight``. With a
    complete inventory where everything named is already gone nothing is saved (``already_retired``);
    a class that could not be read keeps the fact saved, since material may remain there."""
    from ouroboros.task_custody import task_custody_lock
    from ouroboros.task_results import load_task_result
    from ouroboros.task_store import patch_document
    from ouroboros.utils import utc_now_iso

    base = pathlib.Path(root)

    def answer(status: str, stands: bool, *, reason: str = "", retained: Optional[List[str]] = None,
               retired: Optional[List[str]] = None, unknown: Optional[List[str]] = None) -> OwnerKeepResponse:
        return {"task_id": task_id, "keep": stands, "status": status,  # type: ignore[typeddict-item]
                "reason": reason, "retained": retained or [], "retired": retired or [], "unknown": unknown or []}

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
        present = {place: _present(base, task_id, place, pins) for place in places}
        retained = [place for place in places if present[place]]
        retired = [place for place in places if present[place] is False]
        if None in present.values() and "private_copies" not in unknown:
            unknown.append("private_copies")
        if keep and retired and not retained and not unknown:
            return answer("already_retired", stands, retired=retired)
        if keep and not stands:
            fact = {"at": utc_now_iso(), "scope": SCOPE_TASK, "source": source,
                    **({"note": note[:_NOTE_MAX]} if note else {})}
            patch_document(base, task_id, lambda current: {**current, OWNER_KEEP: fact})
        elif not keep and stands:
            patch_document(base, task_id, lambda current: {k: v for k, v in current.items() if k != OWNER_KEEP})
    return answer(("partly_retired" if retired else "kept") if keep else "released", keep,
                  retained=retained, retired=retired, unknown=unknown)

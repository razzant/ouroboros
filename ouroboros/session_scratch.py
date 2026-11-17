"""Session scratch custody: who holds each recorded scratch directory, why, and since when.

A writable session's short scratch (``workspace_preflight._session_scratch``, an ``obs-*`` directory
under ``test_environment.session_scratch_parent()``) is owned through one ownership record,
``session_scratch_owners/<key>/scratch.json``, kept beside its holder: a task's drive for the
scratch its snapshot or in-place starts reuse, a reviewer seat's home for the seat's own. The
holder's end releases the scratch, through the custody trash, whose remover deletes it off the
serving process; nothing scans the temp root, follows a link or sweeps by age. A scratch its
producer marked retained (``test_environment.RETENTION_MARKER``) keeps its record as a visible
hold, ``producer_retained``, in its own trash entry until the marker is gone; a kept service's
declared outputs there no complete capture consumed hold it the same way (``kept_service_capture_owed``);
new records name an ``obs-*`` container with an ordinary nonce marker beside its empty ``w/`` workspace.
The exact recorded address and marker identify it across reboot or whole-folder restoration, not device/inode;
restored contents inherit its holder and cleanup. Legacy records keep their original device/inode check and
are never upgraded. Unverified identity is held, never repaired. This module owns allocation and identity;
``workspace_preflight`` serializes preparation and runs the custody pass over finished tasks.
"""
from __future__ import annotations

import contextlib
import json
import logging
import os
import pathlib
import stat
import tempfile
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List, Optional, Tuple

from ouroboros import work_domain
from ouroboros.test_environment import SCRATCH_CONTAINER_MARKER as SCRATCH_MARKER, SCRATCH_WORKSPACE

log = logging.getLogger(__name__)

# The hold a producer's retention marker puts on a scratch: the marker's own words are its example.
PRODUCER_RETAINED = "producer_retained"
# A real container whose nonce (or legacy device/inode) cannot be confirmed (``ScratchIdentityMismatch``).
IDENTITY_UNVERIFIED = "identity_unverified"
# A seat-keyed record (``review_session_preparation.seat_service_directories``) is held by its seat's copy.
SEAT_OPEN = "seat_open"
# The holders a record may sit beside; ``trash`` and ``staging`` are the handoff on its way out.
HOLDERS = ("task", "seat", "trash", "staging")


def scratch_parents() -> Tuple[pathlib.Path, ...]:
    """The parents a recorded scratch may sit under: the session parent, and the host's default
    temp parent that records made before ``session_scratch_parent`` existed still name."""
    from ouroboros.test_environment import default_temp_parent, session_scratch_parent

    parents = [parent.resolve() for parent in (session_scratch_parent(), default_temp_parent())]
    try:  # a managed session's scratch is the work domain's, under its staging root (``workspace_preflight``)
        parents.append(pathlib.Path(work_domain.staging_path("scratch")))
    except work_domain.WorkDomainRefused:
        pass
    return tuple(dict.fromkeys(parents))


def _overlaps(path: pathlib.Path, execution: pathlib.Path) -> bool:
    return path == execution or path in execution.parents or execution in path.parents


def work_session_scratch(execution: pathlib.Path) -> pathlib.Path:
    """A managed session's scratch (``workspace_preflight._session_scratch``): a new ``obs-*`` folder under
    ``<staging_root>/scratch``, made by the work domain itself (an existing name refuses, as ``mkdtemp`` would
    retry), apart from the execution tree."""
    import secrets

    parent = pathlib.Path(work_domain.staging_path("scratch"))
    work_domain.make_dirs(parent, mode=0o755)
    for _attempt in range(8):
        path = parent / f"obs-{secrets.token_hex(4)}"
        if _overlaps(path, execution):
            raise ValueError(f"{path} overlaps the execution tree {execution}")
        try:
            work_domain.make_dir(path, mode=0o755)
        except work_domain.WorkDomainUnsettled:
            raise  # an unconfirmed allocation is not an available name to retry
        except OSError:
            continue
        if work_domain.kind(path) != "dir":
            raise ValueError(f"{path} is not a real directory")
        return path
    raise OSError(f"no new session scratch could be made under {parent}")


def work_download_caches(execution: pathlib.Path) -> List[pathlib.Path]:
    """A managed session's shared download caches (``uv``, ``pip``, ``npm``) under the staging root's package
    downloads folder, made by the work domain: the service's own data root is never one the work engine is handed."""
    from ouroboros.workspace_preflight import _CACHE_VARIABLES, PACKAGE_DOWNLOADS_DIRNAME

    caches = []
    for name, _variable in _CACHE_VARIABLES:
        path = pathlib.Path(work_domain.staging_path(PACKAGE_DOWNLOADS_DIRNAME, name))
        if _overlaps(path, execution):
            raise ValueError(f"{path} overlaps the execution tree {execution}")
        work_domain.make_dirs(path)
        caches.append(path)
    return caches


class ScratchIdentityMismatch(ValueError):
    """A valid record names a real allowed container, but its identity cannot be confirmed.

    Legacy records compare device/inode; schema 2 requires its nonce marker, with no inode fallback.
    """

    def __init__(self, record: pathlib.Path, path: pathlib.Path, recorded: Any, found: Any, *, basis: str = "dev, ino"):
        self.record, self.path, self.recorded, self.found = record, path, recorded, found
        self.basis = basis
        kind = "" if basis == "dev, ino" else f"{basis} "
        super().__init__(f"{path} is not the recorded scratch directory: recorded {kind}identity {recorded}, found {found}"
                         f" ({record})")


def _scratch_record(record: pathlib.Path) -> Dict[str, Any]:
    if record.is_symlink():
        raise ValueError(f"{record} is not a real ownership record")
    try:
        return json.loads(record.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise OSError(f"scratch ownership record vanished before it was read: {record}") from None


def _marker_matches(path: pathlib.Path, nonce: str) -> str:
    """Empty when the regular nofollow marker contains exactly this nonce, else the observed problem.

    Both paths use existing confined opens: a work path descends from its mapping's anchor, never the
    link/permission fallback of ``work_domain.read_bytes``. Size and read are bounded to the marker's 33 bytes.
    """
    from ouroboros.confined_files import open_regular_file

    expected = (nonce + "\n").encode("ascii")
    marker = path / SCRATCH_MARKER
    try:
        with contextlib.ExitStack() as held:
            spelling = work_domain.work_spelling(marker)
            if spelling:
                directory, name = held.enter_context(work_domain._confined_parent(spelling))
                stream, seen = directory.open(name)
            else:
                stream, seen = open_regular_file(path, SCRATCH_MARKER)
            with stream:
                if seen.st_size != len(expected):
                    return "wrong size"
                return "" if stream.read(len(expected) + 1) == expected else "different nonce"
    except OSError as exc:
        return f"unreadable ({type(exc).__name__})"


def _recorded_scratch(record: pathlib.Path, row: Dict[str, Any]) -> pathlib.Path:
    if not isinstance(row["path"], str):
        raise ValueError(f"{record} records no scratch path")
    path = pathlib.Path(row["path"])
    version = row.get("schema_version", 1)
    if type(version) is not int or version not in (1, 2):
        raise ValueError(f"{record} has an unknown scratch schema")
    recorded = row.get("nonce") if version == 2 else row["identity"]
    if version == 2:
        if not (isinstance(recorded, str) and len(recorded) == 32
                and all(part in "0123456789abcdef" for part in recorded)):
            raise ValueError(f"{record} records no scratch nonce")
    elif not (isinstance(recorded, list) and len(recorded) == 2 and all(type(part) is int for part in recorded)):
        raise ValueError(f"{record} records no scratch path and device/inode identity")
    if path.parent not in scratch_parents() or not path.name.startswith("obs-"):
        raise ValueError(f"{record} does not name a session scratch directory")
    seen = work_domain.lstat(path)  # a missing temporary directory is a gap, never silently re-created
    if not stat.S_ISDIR(seen.st_mode) or (version == 2 and getattr(seen, "st_file_attributes", 0) & 0x400):
        raise ValueError(f"{path} is not the recorded scratch directory")
    if version == 2:
        problem = _marker_matches(path, recorded)
        if problem:
            raise ScratchIdentityMismatch(record, path, recorded, problem, basis="marker")
    elif not _legacy_identity_matches(seen, recorded):
        raise ScratchIdentityMismatch(record, path, recorded, [seen.st_dev, seen.st_ino])
    return path


def _legacy_identity_matches(seen: Any, recorded: List[int], *, windows: bool = os.name == "nt") -> bool:
    """A schema-1 ``[st_dev, st_ino]``. On Windows CPython 3.12+ reports the volume's 64-bit serial; a record
    made under 3.10 holds its low 32 bits, the serial that Python reported then."""
    if [seen.st_dev, seen.st_ino] == recorded:
        return True
    return windows and 0 <= recorded[0] < 1 << 32 and [seen.st_dev & 0xFFFFFFFF, seen.st_ino] == recorded


def recorded_scratch(record: pathlib.Path) -> pathlib.Path:
    """Read-only identity check returning the recorded container, never its granted workspace.

    Invalid records/paths raise ``ValueError``, unreadable records ``OSError``, a real container with
    unverified identity ``ScratchIdentityMismatch``. Only a missing container raises ``FileNotFoundError``:
    a missing/linked/special/oversized nonce marker never means the container is absent. No record or marker
    is repaired, touched or upgraded, even when a legacy record still matches.
    """
    return _recorded_scratch(record, _scratch_record(record))


def _workspace(container: pathlib.Path) -> pathlib.Path:
    """The empty workspace on allocation, or the same real workspace on reuse; no link/junction is granted."""
    path = container / SCRATCH_WORKSPACE
    try:
        seen = work_domain.lstat(path)
    except FileNotFoundError:
        if work_domain.is_work(path):
            work_domain.make_dir(path)
        else:
            path.mkdir()
        seen = work_domain.lstat(path)
    if not stat.S_ISDIR(seen.st_mode) or getattr(seen, "st_file_attributes", 0) & 0x400:
        raise ValueError(f"{path} is not a real scratch workspace")
    return path


def prepare_session_scratch(record: pathlib.Path, execution: pathlib.Path) -> pathlib.Path:
    """Under the caller's preparation lock: reuse its recorded workspace or allocate a schema-2 container.

    Publish only after container -> marker -> empty workspace. Failed preparation rolls back only those
    known names of this fresh, ungranted allocation, never a previous container. A missing legacy directory
    may allocate anew, but a mismatch never upgrades or adopts it. Restoration with the same nonce at the
    same address deliberately inherits the old holder and cleanup, including files added after restoration.
    """
    from ouroboros.test_environment import git_ancestor, new_scratch_identity, session_scratch_parent, write_scratch_marker
    from ouroboros.utils import atomic_write_json
    from ouroboros.workspace_preflight import _owned_directory

    if os.path.lexists(record):
        row = _scratch_record(record)
        try:
            path = _recorded_scratch(record, row)
        except FileNotFoundError:
            pass  # OS aging may have removed it; a literal retry never prepares again
        else:
            if git_ancestor(path) is not None or _overlaps(path, execution):
                raise ValueError(f"{path} overlaps a Git working tree or the execution root")
            return _workspace(path) if row.get("schema_version") == 2 else path
    managed = work_domain.managed()
    parent = pathlib.Path(work_domain.staging_path("scratch")) if managed else session_scratch_parent().resolve()
    if not managed and git_ancestor(parent) is not None:
        raise ValueError(f"{parent} is inside a Git working tree")
    path = work_session_scratch(execution) if managed else pathlib.Path(tempfile.mkdtemp(prefix="obs-", dir=parent))
    row = new_scratch_identity(path)
    marker, nonce = path / SCRATCH_MARKER, row["nonce"]
    try:
        if managed:
            # A work user may differ from the service UID. This new, non-secret marker must be readable
            # through the service's confined open too, without a work-domain cat/link fallback.
            work_domain.write_file(marker, (nonce + "\n").encode("ascii"), mode=0o644)
        else:
            _owned_directory(parent, (path.name,), execution)
            write_scratch_marker(path, nonce)
        workspace = _workspace(path)
        if managed and (problem := _marker_matches(path, nonce)):
            raise ValueError(f"new scratch marker is not readable by its owner: {problem}")
        atomic_write_json(record, row, fsync=True)
    except work_domain.WorkDomainUnsettled:
        raise  # its writer may still be using this new container; custody must settle it, not a rollback
    except Exception:
        # No session received this new allocation: remove only our two known children, then the empty container.
        if managed:
            if work_domain.kind(path / SCRATCH_WORKSPACE) == "dir":
                work_domain.remove_empty_dir(path / SCRATCH_WORKSPACE)
            work_domain.remove(marker)
            work_domain.remove_empty_dir(path)
        else:
            with contextlib.suppress(FileNotFoundError):
                (path / SCRATCH_WORKSPACE).rmdir()
            with contextlib.suppress(FileNotFoundError):
                marker.unlink()
            path.rmdir()
        raise
    return workspace


def identity_hold(record: pathlib.Path, exc: ScratchIdentityMismatch) -> Dict[str, Any]:
    """The ``identity_unverified`` hold on an unconfirmed marker or legacy device/inode: the scratch
    and its record stay as they are (another run may still use that directory), since the record's time.
    The example leads with what a bounded hold note must keep: the scratch, both identities, then the
    record's key (its holder entry is renamed by the handoff; ``scratch_records`` names the file)."""
    since = None
    with contextlib.suppress(OSError):
        since = record.lstat().st_mtime
    example = f"{exc.path} [{exc.basis}] {exc.recorded} != {exc.found}; record {record.parent.name}"
    return {"class": IDENTITY_UNVERIFIED, "example": example, "since": since,
            "recorded_identity": exc.recorded, "found_identity": exc.found}


def has_record(owner: pathlib.Path) -> bool:
    """Whether the ownership folder OWNER holds its ``scratch.json``. Only an absent one is an uncompleted
    preparation: one that cannot be looked at is ``OSError``, never absent (a forced removal of its folder
    would delete it unread)."""
    try:
        os.lstat(owner / "scratch.json")
    except FileNotFoundError:
        return False
    return True


def owner_records(holder: pathlib.Path) -> Iterator[pathlib.Path]:
    """The ownership folders of HOLDER (a drive or a seat home): each a real directory under its
    ``session_scratch_owners/``; a link anywhere in that layout, or an owner that is not a directory,
    is ``OSError`` (nothing is read through it). A folder without ``scratch.json`` is an uncompleted
    preparation and is skipped (``has_record``)."""
    from ouroboros.workspace_preflight import SESSION_SCRATCH_DIRNAME

    if holder.is_symlink():
        raise OSError(f"linked scratch ownership drive: {holder}")
    owners = holder / SESSION_SCRATCH_DIRNAME
    if owners.is_symlink():
        raise OSError(f"linked scratch ownership directory: {owners}")
    if not owners.is_dir():
        return
    for owner in sorted(owners.iterdir()):
        if owner.is_symlink() or not owner.is_dir():
            raise OSError(f"invalid scratch owner: {owner}")
        if has_record(owner):
            yield owner


def _nested_drives(drive: pathlib.Path) -> Iterator[pathlib.Path]:
    """The existing child layouts inside a drive (never arbitrary folders): ``task_drives/<id>`` and
    ``state/headless_tasks/<id>/data``, each a real directory reached through no link."""
    for relative, suffix in (("task_drives", ()), ("state/headless_tasks", ("data",))):
        children = drive / relative
        if (children.is_dir() and not children.is_symlink()
                and all(not parent.is_symlink() for parent in children.parents
                        if parent != drive and drive in parent.parents)):
            for child in sorted(children.iterdir()):
                nested = child.joinpath(*suffix)
                if child.is_dir() and not child.is_symlink() and nested.is_dir() and not nested.is_symlink():
                    yield nested


def retention_hold(path: pathlib.Path) -> Optional[Dict[str, Any]]:
    """The ``producer_retained`` hold on a scratch, or None: the first marker beneath it names the
    reason (its first line; an unreadable subtree counts as one), and when it was written."""
    from ouroboros.test_environment import RETENTION_MARKER, retention_markers

    if work_domain.is_work(path):  # walked by held descriptors, its marker read confined; an unlistable folder holds
        markers: List[pathlib.Path] = []
        for folder, _dirs, files in work_domain.walk(path, onerror=lambda exc: markers.append(pathlib.Path(exc.filename))):
            if RETENTION_MARKER in files:
                markers.append(pathlib.Path(folder) / RETENTION_MARKER)
    else:
        markers = retention_markers(path)
    if not markers:
        return None
    marker, reason, since = markers[0], "unreadable subtree", None
    with contextlib.suppress(OSError):
        since = work_domain.lstat(marker).st_mtime
    if work_domain.kind(marker) == "file":
        with contextlib.suppress(OSError, UnicodeError):
            reason = (work_domain.read_text(marker).strip().splitlines() or ["retained"])[0]
    return {"class": PRODUCER_RETAINED, "example": f"{path}: {reason}", "since": since}


def _capture_owed_hold(record: pathlib.Path, path: pathlib.Path, owed: Any) -> Optional[Dict[str, Any]]:
    """The ``kept_service_capture_owed`` hold on a scratch holding a kept launch's declared outputs no complete capture
    consumed, since its record's time, or None; ``owed`` is the pass's one selection (``owed_declarations``)."""
    from ouroboros.tools.kept_service_outputs import owed_output_hold

    why = owed_output_hold(None, path, owed=owed) if owed else ""
    if not why:
        return None
    since = None
    with contextlib.suppress(OSError):
        since = record.lstat().st_mtime
    return {"class": why, "example": f"{path}: a kept service's declared outputs are not captured yet", "since": since}


def scratch_capture_owed(owner: pathlib.Path, owed: Any) -> str:
    """Why the scratch OWNER's record names may not be released yet: ``kept_service_capture_owed`` while it holds a
    kept launch's uncaptured declared outputs, else ``""`` (an unreadable record names nothing here; the trash pass
    keeps it). ``owed``: the pass's one selection, so a pass over many records reads the declarations once."""
    if not owed:
        return ""
    try:
        path = str(json.loads((owner / "scratch.json").read_text(encoding="utf-8"))["path"])
    except (OSError, ValueError, KeyError, TypeError):
        return ""
    from ouroboros.tools.kept_service_outputs import owed_output_hold

    return owed_output_hold(None, path, owed=owed)


def release_session_scratch(drive: pathlib.Path, *, owed: Any = ()) -> List[Dict[str, Any]]:
    """Delete the scratch every ownership record under DRIVE (and its nested child layouts) names:
    the drive's custody retired (``task_custody.remove_custody_trash`` calls this on each trash entry,
    the handoff entries of single records included, always with its pass's ``owed`` selection).

    Returns the records whose scratch stays in place, each with ``owner`` (the record's folder) and
    ``path``: a kept service's outputs still owed a capture (``_capture_owed_hold``), a producer retention
    marker's (``retention_hold``, #2708) and a real ``obs-*`` container whose identity its valid record cannot
    confirm (``identity_hold``; neither it nor its markers are touched). The caller keeps each as a
    visible hold, never deleting its record with the entry; the other records are still released. An
    unreadable or malformed record, a foreign, linked or non-directory path, or an incomplete removal is
    ``OSError``: the entry and its pointer wait for the next pass. No temp-root scan, arbitrary symlink
    following or independent sweeper."""
    from ouroboros.force_remove import force_rmtree

    retained: List[Dict[str, Any]] = []
    for owner in owner_records(drive):
        record = owner / "scratch.json"
        try:
            path = recorded_scratch(record)
        except FileNotFoundError:  # a prior pass, or a person, already removed this scratch (never the record)
            continue
        except ScratchIdentityMismatch as exc:  # ownership unproven: kept, and so is its record, never deleted
            retained.append({**identity_hold(record, exc), "owner": owner, "path": str(exc.path)})
            continue
        except (ValueError, KeyError, TypeError) as exc:
            raise OSError(f"invalid scratch ownership record: {record}") from exc
        hold = _capture_owed_hold(record, path, owed) or retention_hold(path)
        if hold is not None:
            retained.append({**hold, "owner": owner, "path": str(path)})  # another's to keep: never deleted here
            continue
        if work_domain.is_work(path):  # a work scratch is removed by the work domain, its identity checked above
            work_domain.remove_tree(path)
            if work_domain.exists(path):
                raise OSError(f"session scratch removal incomplete: {path}")
            continue
        force_rmtree(path)
        if os.path.lexists(path):
            raise OSError(f"session scratch removal incomplete: {path}")
    for nested in _nested_drives(drive):
        retained += release_session_scratch(nested, owed=owed)
    return retained


def release_trash_entry(data_root: Any, entry: pathlib.Path, *, owed: Any = None) -> Tuple[bool, List[Dict[str, Any]]]:
    """Release the scratch one custody trash ENTRY's records name (its ``data/`` and nested layouts too) and
    hand each record whose scratch stays (owed a capture, a retention marker's, or an ``identity_unverified`` one:
    only its record file moves, never the scratch) to an entry of its own (``workspace_preflight._retire_record``)
    unless ENTRY is such a handoff already, which then stays: ``(stays, the retained rows)``. Every failure
    raises, so ``task_custody.remove_custody_trash`` keeps the entry and every record still in it; a failed
    handoff carries the retained rows (``retained``), whose scratch stands whether its record moved or not.
    ``owed``: the pass's one selection (``owed_declarations``); read here when the caller passes none."""
    from ouroboros.tools.kept_service_outputs import owed_declarations
    from ouroboros.workspace_preflight import _RELEASE_MARK, _retire_record

    selection = owed_declarations(data_root) if owed is None else owed
    retained: List[Dict[str, Any]] = []
    if entry.is_dir() and not entry.is_symlink():
        retained = release_session_scratch(entry, owed=selection)
        if (entry / "data").is_dir() and not (entry / "data").is_symlink():
            retained += release_session_scratch(entry / "data", owed=selection)
    stays = bool(retained) and _RELEASE_MARK in entry.name
    try:
        for row in [] if stays else retained:
            if not _retire_record(pathlib.Path(data_root), entry.name.split("+", 1)[0].rsplit("-", 1)[0], row["owner"]):
                raise OSError(f"retained scratch record not handed on: {row['owner']}")
    except Exception as exc:
        exc.retained = retained  # an earlier record may already sit in its own entry, unseen by this pass
        raise
    return stays, retained


def release_seat_scratch(data_root: Any, task_id: str, home: pathlib.Path) -> Dict[str, Any]:
    """Hand the scratch records a seat home holds to the custody trash: the seat's copy is released
    (its run ended, on the seat's own proof), so the scratch it served is over too, whatever the
    seat's task is doing (#2706) and whether that task exists (#2707). Called under the seat lock
    by the two places that release the copy. ``{"released": [owners], "errors": [...]}``; a record
    whose handoff fails stays in the home and is released when the home is read again without its
    copy (``review_checking_prune``) or retires whole into the trash (``remove_custody_trash``)."""
    from ouroboros.workspace_preflight import SESSION_SCRATCH_DIRNAME, _retire_record

    root, report = pathlib.Path(data_root), {"released": [], "errors": []}
    try:
        owners = list(owner_records(home))
    except OSError as exc:
        report["errors"].append(f"{home.name}: {type(exc).__name__}: {exc}")
        return report
    for owner in owners:
        try:
            if _retire_record(root, task_id, owner):
                report["released"].append(str(owner))
        except OSError as exc:
            log.warning("Seat scratch record %s was not released", owner, exc_info=True)
            report["errors"].append(f"{owner}: {type(exc).__name__}: {exc}")
    with contextlib.suppress(OSError):
        (home / SESSION_SCRATCH_DIRNAME).rmdir()  # empty once every record went; a survivor keeps it
    return report


def seat_hold(data_root: Any, key: str) -> str:
    """Why a seat-keyed record (``seat-<home>``) still holds its scratch: ``seat_open`` while the
    seat's copy is present (a run may still execute there), ``""`` once the copy was released or
    the home is gone, the seat's own proof. Records made before seats held their own records sit in
    task drives and are judged this way by ``workspace_preflight.release_finished_scratch``."""
    from ouroboros.review_checking import _checking_area_present
    from ouroboros.unindexed import unindexed
    from ouroboros.workspace_preflight import SEAT_SCRATCH_KEY_PREFIX

    home = pathlib.Path(data_root) / "state" / "review_checkouts" / "seats" / key[len(SEAT_SCRATCH_KEY_PREFIX):]
    if not home.is_dir() or home.is_symlink():
        return ""
    return SEAT_OPEN if _checking_area_present(unindexed(home, "area")) else ""


def task_hold(data_root: Any, task_id: str, live: Any) -> str:
    """Why a task still holds its snapshot or in-place scratch, ``""`` when it is proven over and no
    delegated run is open. The decision is ``subagent_worktrees._task_proven_over`` and
    ``workspace_preflight._runs_still_open``; the typed word names which of their facts failed
    (the order they read them in): ``task_not_terminal``, ``task_live``, ``liveness_unknown``,
    ``store_busy`` (the probe met a contended store: this task's hold, never the pass's end),
    ``owner_processes_live``, ``run_open``, ``custody_unreadable``."""
    from ouroboros.body_candidate import _recorded_terminal
    from ouroboros.delegate_custody_current import task_snapshot
    from ouroboros.store import StoreBusy
    from ouroboros.subagent_worktrees import _task_proven_over
    from ouroboros.workspace_preflight import _runs_still_open

    root = pathlib.Path(data_root)
    if not _task_proven_over(task_id, root, live):
        if not _recorded_terminal(task_id, root):
            return "task_not_terminal"
        try:
            probe = live(task_id) if live is not None else None
        except StoreBusy:
            return "store_busy"
        return "task_live" if probe is True else "liveness_unknown" if probe is not False else "owner_processes_live"
    if _runs_still_open(root, task_id):
        try:
            task_snapshot(root, task_id)
        except Exception:
            return "custody_unreadable"
        return "run_open"
    return ""


def _iso(stamp: Optional[float]) -> str:
    return datetime.fromtimestamp(stamp, tz=timezone.utc).isoformat(timespec="seconds") if stamp else ""


def aggregate_holds(rows: List[Tuple[str, str, Optional[float]]]) -> List[Dict[str, Any]]:
    """``[{class, count, example, since}]`` from ``(class, example, since_timestamp)`` rows, one per
    class in name order: the first example, cut to the engine's bound as ``maintenance_jobs.Holds`` cuts a
    note (a producer's marker line can be any length), and the earliest since (the engine's ``GcHold``
    shape, which the maintenance job rows, Storage and the mind's ``storage`` fact render alike)."""
    from ouroboros.maintenance_jobs import HOLD_EXAMPLE_CHARS

    holds: Dict[str, Dict[str, Any]] = {}
    for kind, example, since in rows:
        hold = holds.setdefault(kind, {"class": kind, "count": 0, "example": str(example or "")[:HOLD_EXAMPLE_CHARS],
                                       "since": since})
        hold["count"] += 1
        if since is not None and (hold["since"] is None or since < hold["since"]):
            hold["since"] = since
    return [{**hold, "since": _iso(hold["since"])} for _kind, hold in sorted(holds.items())]


def _holders(root: pathlib.Path) -> Iterator[Tuple[str, str, pathlib.Path]]:
    """``(holder kind, task id, drive or home)`` for every place a record may sit, through no link."""
    from ouroboros.task_custody import _STAGING_DIR, _TRASH_DIR

    def listing(folder: pathlib.Path) -> List[pathlib.Path]:
        try:
            return sorted(entry for entry in folder.iterdir() if entry.is_dir() and not entry.is_symlink()) \
                if folder.is_dir() and not folder.is_symlink() else []
        except OSError:
            return []

    for drive in listing(root / "task_drives"):
        yield "task", drive.name, drive
    for child in listing(root / "state" / "headless_tasks"):
        if (child / "data").is_dir() and not (child / "data").is_symlink():
            for drive in listing(child / "data" / "task_drives"):
                yield "task", drive.name, drive
    for home in listing(root / "state" / "review_checkouts" / "seats"):
        if not home.name.startswith("."):
            with contextlib.suppress(OSError, ValueError, AttributeError):
                marker = json.loads((home / "checking.json").read_text(encoding="utf-8"))
                yield "seat", str(marker.get("task_id") or ""), home
    for kind, folder in (("trash", root / _TRASH_DIR), ("staging", root / _STAGING_DIR)):
        for entry in listing(folder):
            task_id = entry.name.split("+", 1)[0].rsplit("-", 1)[0]
            yield kind, task_id, entry
            if (entry / "data").is_dir() and not (entry / "data").is_symlink():
                yield kind, task_id, entry / "data"
            for nested in _nested_drives(entry):
                yield kind, nested.parent.name if nested.name == "data" else nested.name, nested


def scratch_records(data_root: Any, live: Any = None, *, include_holds: bool = True) -> List[Dict[str, Any]]:
    """Every session scratch ownership record of the install, for Storage's ``session_scratch`` place:
    ``holder`` (one of ``HOLDERS``), ``task_id``, ``key``, ``record`` (the file), ``path`` (the scratch,
    validated as ``recorded_scratch`` does: ``valid`` False and ``problem`` when it names a foreign
    path, an unconfirmed identity or a link; ``missing`` when the container is gone), ``hold`` (the typed
    reason it is kept: ``task_hold`` words, ``seat_open``, ``owner_keep``, ``producer_retained``,
    ``identity_unverified`` with ``recorded_identity``/``found_identity``, never ``valid``, any holder;
    ``""`` when its release is due or under way) and ``since`` (ISO, the record's time, or the
    marker's for a producer hold). ``live`` is the supervisor's liveness probe; without one a task's
    liveness reads unknown. Records only, no temp-root scan; an unreadable holder or record is one row.
    ``include_holds=False`` validates paths for counting without reading holder/retention proofs."""
    from ouroboros.owner_keep import is_kept
    from ouroboros.tools.kept_service_outputs import owed_declarations, owed_output_hold
    from ouroboros.workspace_preflight import SEAT_SCRATCH_KEY_PREFIX

    root, rows, task_holds = pathlib.Path(data_root), [], {}
    owed = owed_declarations(root) if include_holds else ()  # one read for every record of this view
    for kind, task_id, holder in _holders(root):
        try:
            owners = list(owner_records(holder))
        except OSError as exc:
            rows.append({"holder": kind, "task_id": task_id, "key": "", "record": str(holder), "path": "",
                         "valid": False, "missing": False, "problem": f"{type(exc).__name__}: {exc}",
                         "hold": "", "since": ""})
            continue
        for owner in owners:
            record, row = owner / "scratch.json", {"holder": kind, "task_id": task_id, "key": owner.name}
            row.update(record=str(record), path="", valid=True, missing=False, problem="", hold="")
            with contextlib.suppress(OSError):
                row["since"] = _iso(record.lstat().st_mtime)
            row.setdefault("since", "")
            try:
                path = recorded_scratch(record)
            except FileNotFoundError:
                row.update(missing=True)
                with contextlib.suppress(OSError, ValueError, KeyError, TypeError):
                    row["path"] = str(json.loads(record.read_text(encoding="utf-8"))["path"])
                rows.append(row)
                continue
            except ScratchIdentityMismatch as exc:  # never counted or offered: its ownership is unproven
                hold = identity_hold(record, exc)
                row.update(valid=False, problem=f"{type(exc).__name__}: {exc}", path=str(exc.path), hold=hold["class"],
                           example=hold["example"], recorded_identity=exc.recorded, found_identity=exc.found)
                rows.append(row)
                continue
            except (OSError, ValueError, KeyError, TypeError) as exc:
                row.update(valid=False, problem=f"{type(exc).__name__}: {exc}")
                rows.append(row)
                continue
            row["path"] = str(path)
            if not include_holds:
                rows.append(row)
                continue
            hold = retention_hold(path) if kind in ("trash", "staging") else None
            if hold is not None:
                row.update(hold=hold["class"], since=_iso(hold["since"]) or row["since"], example=hold["example"])
            elif kind == "seat":
                row["hold"] = seat_hold(root, owner.name)
            elif kind == "task":
                if owner.name.startswith(SEAT_SCRATCH_KEY_PREFIX):
                    row["hold"] = seat_hold(root, owner.name)
                else:
                    if task_id not in task_holds:
                        task_holds[task_id] = task_hold(root, task_id, live)
                    row["hold"] = task_holds[task_id]
            if kind in ("task", "seat") and not row["hold"] and task_id and is_kept(root, task_id):
                row["hold"] = "owner_keep"
            if owed and not row["hold"]:
                row["hold"] = owed_output_hold(root, path, owed=owed)
            rows.append(row)
    return rows


__all__ = [
    "HOLDERS",
    "IDENTITY_UNVERIFIED",
    "PRODUCER_RETAINED",
    "ScratchIdentityMismatch",
    "SEAT_OPEN",
    "aggregate_holds",
    "has_record",
    "identity_hold",
    "owner_records",
    "prepare_session_scratch",
    "recorded_scratch",
    "release_seat_scratch",
    "release_session_scratch",
    "release_trash_entry",
    "retention_hold",
    "scratch_capture_owed",
    "scratch_records",
    "seat_hold",
    "task_hold",
]


def recorded_scratch_roots(data_root: Any) -> List[str]:
    """The scratch directories the install's records name and that exist, each once: the roots of the
    counted Storage place ``session_scratch`` (``storage.PLACES``). Records that are invalid or whose
    directory is gone name no root (``scratch_records`` reports them); no temp-root scan."""
    roots: List[str] = []
    for row in scratch_records(data_root, include_holds=False):
        if row["valid"] and not row["missing"] and row["path"] and row["path"] not in roots:
            roots.append(row["path"])
    return roots

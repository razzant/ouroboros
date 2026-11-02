"""Dashboard -> Storage: one complete count of every place Ouroboros owns, in a child, saved.

A *place* is a root Ouroboros owns (``PLACES``: the data folder's parts, the engine home inside
it, the configured working-copy, project and Deliverables roots, the installed program); a
*group* is one of the five bar segments a place belongs to. The count is one complete streaming
metadata walk of every place, with no time, entry or depth bound, run only when the owner asks
(``POST /api/storage/measure``) in a supervised child process at low CPU and disk priority
(``python -m ouroboros.storage``), with progress and Stop (``cancel`` kills the child). Only a
walk that ended by itself writes ``state/storage_measurement.json`` (the latest and the previous
finished count, so the gauge can show growth); a cancelled, failed or killed count never touches
it. ``StorageCount`` is the server's owner: it spawns the child through the maintenance-job seam,
streams its lines, loads the saved file and re-reads it when its identity changes, and never
walks on a server thread. ``gateway.storage`` mounts the routes.

The walk reads directory entries and ``lstat`` sizes only, never file contents. Depth-first, it
keeps open only the folders on its current path; a link or junction is never counted and never
followed (directory-relative no-follow opens where the platform has them, a by-path recheck just
before each listing elsewhere); a vanished entry is empty; an unreadable entry adds one to
``unreadable`` and the walk goes on, so a finished count with unreadable items is still the
headline. Bytes are allocated (``st_blocks``) where the platform reports them, logical otherwise
(``basis`` says which; both are recorded); a multiply linked file counts once by its
``(device, inode)`` identity, which Windows listings lack, so there every linked file is asked
by path (the honest total is the point; the cost is time inside a low-priority child). Every
root's identity is collected first and a walk skips a folder that is another place's root, so
the places sum to the headline whatever the roots nest like. Pacing is physical: one
millisecond of sleep per 512 ``stat`` calls. A descriptor limit a very deep tree exhausts makes
that subtree unreadable, counted as such; the walk goes on.

``--size-paths`` sizes only the directories named on stdin with the same walker (the free-space
preview's sizing, ``storage_free``): one JSON string per line in, one JSON line per path out, so
the pipe carries ASCII whatever the platform's code page; nothing saved.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import pathlib
import shutil
import stat
import sys
import threading
import time
from typing import Any, Callable, Dict, FrozenSet, List, Literal, Optional, Set, Tuple

from ouroboros.utils import atomic_write_json, read_json_dict, utc_now_iso

try:  # Python 3.11+
    from typing import NotRequired, TypedDict  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - CI supports Python 3.10.
    from typing_extensions import NotRequired, TypedDict  # type: ignore[assignment]

from ouroboros.storage_free import StorageFree  # the tray's envelope, homed with its owner; resolved for the contract schema
from ouroboros.storage_removers import StorageRemover  # each place's removers, homed with their reader

log = logging.getLogger(__name__)

PURPOSE = "storage"  # the custody purpose and the module the child runs
SCHEMA_VERSION = 1
MEASUREMENT_FILE = pathlib.Path("state") / "storage_measurement.json"
ENGINE_DIR = "claudexor"  # the owned engine home's name (claudexor_daemon.owned_config_dir)
GROUPS: Tuple[str, ...] = ("records", "engine", "work", "temporary", "other")
# (id, group, data-relative roots or a config getter name): walk order and list order.
PLACES: Tuple[Tuple[str, str, Any], ...] = (
    ("model_calls", "records", (("observability",),)),
    ("engine", "engine", ((ENGINE_DIR,),)),
    ("task_results", "work", (("task_results",),)),
    ("archives", "work", (("archive",),)),
    ("memory_skills", "work", (("memory",), ("projects",), ("skills",), ("state", "skills"))),
    ("project_folders", "work", "get_subagent_projects_root"),
    ("deliverables", "work", "get_deliverables_root"),
    ("work_folders", "temporary", (("task_drives",), ("state", "headless_tasks"))),
    ("working_copies", "temporary", "get_subagent_worktree_root"),
    ("app_state", "other", (("state",),)),
    ("data_other", "other", ((),)),
    ("program", "other", "REPO_DIR"),
)
_LARGEST_PARTS = 8
# Parts the client reads by name whatever their rank (the engine's three-part line): always listed.
_NAMED_PARTS: Dict[str, Tuple[str, ...]] = {"engine": ("projects", "profiles")}
_PACE_EVERY = 512  # stat calls between one-millisecond sleeps: pacing one disk, not judgment
_PROGRESS_SEC = 0.25
_LINE_BYTES = 65536

# Directory-relative opening that never follows a link; without it the walk falls back to a
# by-path check just before each listing.
_RELATIVE = bool(getattr(os, "O_NOFOLLOW", 0) and getattr(os, "O_DIRECTORY", 0)
                 and {os.open, os.stat} <= os.supports_dir_fd and os.scandir in os.supports_fd)


class StoragePart(TypedDict):
    """One depth-1 entry of a place's roots (files directly in a root form the part ``(files)``)."""

    name: str
    bytes: int
    files: int


class StorageRest(TypedDict):
    """The parts beyond the largest eight, summed; a place's named parts (``_NAMED_PARTS``) are
    listed whatever their rank, never summed here."""

    count: int
    bytes: int


class StoragePlace(TypedDict):
    """One counted place; ``bytes`` is None when none of its present roots could be listed."""

    id: str
    group: str
    roots: List[str]
    present: bool
    same_volume: bool
    bytes: Optional[int]
    logical_bytes: int
    files: int
    unreadable: int
    parts: List[StoragePart]
    rest: StorageRest
    growth_bytes: NotRequired[Optional[int]]  # against the previous finished count, when both have it
    growth_flag: NotRequired[Literal["unreadable"]]  # either count had unreadable items here
    undecided: NotRequired[str]  # why its default folder cannot be chosen (launch_roots): unknown, never empty


class StorageVolume(TypedDict):
    """Free and total space of the data folder's volume, never a machine total."""

    free_bytes: Optional[int]
    total_bytes: Optional[int]
    observed_at: str


class StorageCount(TypedDict):
    """The last finished count (the saved ``latest``) with growth against the one before it."""

    started_at: str
    finished_at: str
    duration_seconds: float
    io_priority: str
    basis: Literal["allocated", "logical"]
    bytes: int
    logical_bytes: int
    files: int
    unreadable: int
    volume: StorageVolume
    places: List[StoragePlace]
    growth_bytes: NotRequired[Optional[int]]
    growth_since: NotRequired[str]
    growth_flag: NotRequired[Literal["unreadable"]]


class StorageCounting(TypedDict):
    """A running count: where it is; ``done`` lists only finished places with a known size."""

    started_at: str
    place: str
    current: str
    places_done: int
    places_total: int
    bytes: int
    files: int
    done: List[StoragePart]


class StorageError(TypedDict):
    at: str
    message: str
    stopped: bool  # the owner stopped it (``cancel``), as opposed to a failure


class StorageResponse(TypedDict):
    """``GET /api/storage``; the measure, cancel and free POSTs answer it too (``started`` on measure
    and free, ``reason`` when a free run did not start). ``count`` is omitted when
    ``?count=<finished_at>`` names the latest finished count; ``free`` is ``storage_free``'s tray."""

    count: NotRequired[Optional[StorageCount]]
    counting: Optional[StorageCounting]
    error: Optional[StorageError]
    volume: StorageVolume
    retention_days: int
    free: "StorageFree"
    removers: Dict[str, List["StorageRemover"]]  # place id -> its removers, read on every answer (``storage_removers``)
    started: NotRequired[bool]
    reason: NotRequired[str]


class StorageUnavailable(Exception):
    """A count that could not run; its text is shown to the owner."""


# ---- the walk -------------------------------------------------------------------------------

def _directory_flags(follow: bool = False) -> int:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    return flags if follow else flags | getattr(os, "O_NOFOLLOW", 0)


def _close(fd: Optional[int]) -> None:
    if fd is not None:
        with contextlib.suppress(OSError):
            os.close(fd)


def _is_reparse(info: os.stat_result) -> bool:
    """A symlink or Windows junction/reparse point is never traversed."""
    if stat.S_ISLNK(info.st_mode):
        return True
    return bool(getattr(info, "st_file_attributes", 0)
                & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))


def _listing_stat(entry: os.DirEntry, directory_fd: Optional[int]) -> os.stat_result:
    """An entry's own metadata, relative to its open parent where the platform allows."""
    if directory_fd is not None:
        return os.stat(entry.name, dir_fd=directory_fd, follow_symlinks=False)
    return entry.stat(follow_symlinks=False)


def _path_stat(path: str) -> os.stat_result:
    """A by-path ``lstat``: Windows listings carry no inode or link count, the file itself does."""
    return os.lstat(path)


def _allocated(info: os.stat_result) -> Optional[int]:
    blocks = getattr(info, "st_blocks", None)
    return None if blocks is None else int(blocks) * 512


def _identity(path: str) -> Optional[Tuple[int, int]]:
    """A directory's ``(device, inode)``, following links (roots are resolved first); None when
    the path is absent, not a directory, or the platform reports no identity."""
    try:
        info = os.stat(path)
    except (OSError, ValueError):
        return None
    if stat.S_ISDIR(info.st_mode) and info.st_ino:
        return (info.st_dev, info.st_ino)
    return None


def _open_root(base: str, parts: Tuple[str, ...]) -> Tuple[str, Optional[int]]:
    """Open one root: BASE (a resolved configured root, followed) then each PART relative to it,
    never through a link: (``"open"`` | ``"absent"`` | ``"error"``, descriptor)."""
    path, fd = base, None
    try:
        if _RELATIVE:
            fd = os.open(base, _directory_flags(follow=True))
        elif not os.path.isdir(base):
            return ("absent" if not os.path.lexists(base) else "error"), None
        for part in parts:
            path = os.path.join(path, part)
            if fd is not None:
                child = os.open(part, _directory_flags(), dir_fd=fd)
                _close(fd)
                fd = child
            else:
                info = _path_stat(path)
                if _is_reparse(info) or not stat.S_ISDIR(info.st_mode):
                    return "error", None
    except FileNotFoundError:
        _close(fd)
        return "absent", None
    except OSError:  # a link, a file or a refusal where a folder should be
        _close(fd)
        return "error", None
    return "open", fd


class _Walk:
    """What one count shares across its places: linked-file identities, the pacing counter and
    the progress clock. ``skip`` holds the identities of folders other places own."""

    def __init__(self, skip: FrozenSet[Tuple[int, int]] = frozenset(), *, seen: Optional[Set[Tuple[int, int]]] = None,
                 progress: Optional[Callable[[str, int, int], None]] = None,
                 clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep) -> None:
        self.skip = skip
        self.seen: Set[Tuple[int, int]] = set() if seen is None else seen
        self.stats = 0
        self._progress = progress
        self._clock, self._sleep = clock, sleep
        self._reported = clock()

    def paced(self) -> None:
        self.stats += 1
        if self.stats % _PACE_EVERY == 0:
            self._sleep(0.001)

    def tick(self, current: str, counted: int, files: int) -> None:
        if self._progress is not None and self._clock() - self._reported >= _PROGRESS_SEC:
            self._reported = self._clock()
            self._progress(current, counted, files)


def _walk_roots(roots: List[Tuple[str, Tuple[str, ...]]], walk: _Walk, named: Tuple[str, ...] = ()) -> Dict[str, Any]:
    """Count every ROOT (``(base, parts)``) completely: allocated and logical bytes, files,
    unreadable entries, and the depth-1 parts (the largest eight, plus NAMED ones whatever their
    rank). A root the platform cannot list at all leaves the place unknown (``bytes`` None) rather
    than zero; an absent root is a truthful empty."""
    counted = logical = files = unreadable = 0
    present = listed = False
    parts: Dict[str, List[int]] = {}
    prefixed = len(roots) > 1
    # Depth-first: only the folders on the current path are open, each as (path, own descriptor
    # or None, its listing, level, the depth-1 part it belongs to).
    frames: List[Tuple[str, Optional[int], Any, int, str]] = []

    def enter(path: str, fd: Optional[int], level: int, part: str) -> None:
        nonlocal unreadable, listed
        try:
            if fd is None:  # the by-path fallback re-checks the folder just before listing it
                info = _path_stat(path)
                if _is_reparse(info) or not stat.S_ISDIR(info.st_mode):
                    unreadable += 1  # it changed into a link or a file while counting
                    return
            listing = os.scandir(fd if fd is not None else path)
        except FileNotFoundError:
            _close(fd)  # removed while counting: it holds nothing now
            return
        except OSError:
            _close(fd)
            unreadable += 1
            return
        listed = True
        frames.append((path, fd, listing, level, part))

    def leave() -> None:
        _path, fd, listing, _level, _part = frames.pop()
        listing.close()
        _close(fd)

    def account(part: str, size: int) -> None:
        total = parts.setdefault(part, [0, 0])
        total[0] += size
        total[1] += 1

    try:
        for base, rel in roots:
            status, fd = _open_root(base, rel)
            if status == "absent":
                continue
            present = True
            if status != "open":
                unreadable += 1
                continue
            root_path = os.path.join(base, *rel)
            root_name = ("/".join(rel) if rel else os.path.basename(base.rstrip(os.sep))) if prefixed else ""
            enter(root_path, fd, 0, "")
            while frames:
                directory, directory_fd, listing, level, part = frames[-1]
                try:
                    entry = next(listing)
                except (StopIteration, FileNotFoundError):
                    leave()
                    continue
                except OSError:
                    unreadable += 1
                    leave()
                    continue
                child = os.path.join(directory, entry.name)
                try:
                    info = _listing_stat(entry, directory_fd)
                    walk.paced()
                    if _is_reparse(info):
                        continue  # a link is never counted and its target never walked
                    if stat.S_ISDIR(info.st_mode):
                        if not info.st_ino:  # a Windows listing has no identity: ask the folder itself
                            info = _path_stat(child)
                            if _is_reparse(info) or not stat.S_ISDIR(info.st_mode):
                                unreadable += 1  # it changed into a link or a file while counting
                                continue
                        if (info.st_dev, info.st_ino) in walk.skip:
                            continue  # another place's root: counted there
                        name = part if level else (f"{root_name}/{entry.name}" if root_name else entry.name)
                        child_fd = (os.open(entry.name, _directory_flags(), dir_fd=directory_fd)
                                    if directory_fd is not None else None)
                        enter(child, child_fd, level + 1, name)
                        continue
                    if not stat.S_ISREG(info.st_mode):
                        continue
                    if info.st_nlink != 1 and not info.st_ino:
                        # Windows listings carry no link count or inode: ask the file itself.
                        info = _path_stat(child)
                        if not stat.S_ISREG(info.st_mode) or _is_reparse(info):
                            continue
                    if info.st_nlink > 1 and info.st_ino:
                        identity = (info.st_dev, info.st_ino)
                        if identity in walk.seen:
                            continue  # a hard link already counted in this walk
                        walk.seen.add(identity)
                    size = _allocated(info)
                    counted += info.st_size if size is None else size
                    logical += info.st_size
                    files += 1
                    account(part if level else (f"{root_name}/(files)" if root_name else "(files)"),
                            info.st_size if size is None else size)
                except FileNotFoundError:
                    continue  # removed while counting: it holds nothing now
                except OSError:
                    unreadable += 1  # includes a folder swapped for a link before it opened
                walk.tick(part, counted, files)
    finally:
        while frames:
            leave()
    ranked = sorted(parts.items(), key=lambda item: (-item[1][0], item[0]))
    shown = ranked[:_LARGEST_PARTS] + [item for item in ranked[_LARGEST_PARTS:] if item[0] in named]
    rest = [item for item in ranked[_LARGEST_PARTS:] if item[0] not in named]
    return {
        "present": present,
        "bytes": None if present and unreadable and not listed else counted,
        "logical_bytes": logical,
        "files": files,
        "unreadable": unreadable,
        "parts": [{"name": name, "bytes": total[0], "files": total[1]} for name, total in shown],
        "rest": {"count": len(rest), "bytes": sum(total[0] for _name, total in rest)},
    }


# ---- the count ------------------------------------------------------------------------------

def place_roots(data_root: Any) -> List[Dict[str, Any]]:
    """The place table resolved for DATA_ROOT: ``{"id", "group", "roots": [(base, parts)]}`` in
    walk order; configured roots come from ``config`` getters, never from the client. A default
    work root that cannot be chosen carries the refusal as ``undecided`` and no root to walk."""
    from ouroboros import config
    from ouroboros.launch_roots import WorkRootUnavailable

    data = os.path.realpath(os.fspath(data_root))
    rows = []
    for place_id, group, spec in PLACES:
        undecided = ""
        if isinstance(spec, str):
            configured = getattr(config, spec)
            try:
                configured = configured() if callable(configured) else configured
            except WorkRootUnavailable as refusal:  # counted as unknown, never guessed
                configured, undecided = None, str(refusal)
            roots = [(os.path.realpath(os.fspath(configured)), ())] if configured else []
        else:
            roots = [(data, tuple(parts)) for parts in spec]
        rows.append({"id": place_id, "group": group, "roots": roots, **({"undecided": undecided} if undecided else {})})
    return rows


def _volume(path: Any) -> StorageVolume:
    try:
        usage = shutil.disk_usage(os.fspath(path))
        free, total = usage.free, usage.total
    except OSError:
        free = total = None
    return {"free_bytes": free, "total_bytes": total, "observed_at": utc_now_iso()}


def measure(data_root: Any, *, places: Optional[List[Dict[str, Any]]] = None, io_priority: str = "unchanged",
            progress: Optional[Callable[[Dict[str, Any]], None]] = None,
            clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep) -> StorageCount:
    """One complete count of every place (module docstring); the record ``save_count`` keeps.
    PROGRESS, when given, receives a ``StorageCounting``-shaped dict at most every 250 ms."""
    root = os.path.realpath(os.fspath(data_root))
    if not os.path.isdir(root):
        raise StorageUnavailable("The data folder could not be found.")
    table = places if places is not None else place_roots(root)
    began, started_at = clock(), utc_now_iso()
    try:
        volume_device = os.stat(root).st_dev
        basis: Literal["allocated", "logical"] = "allocated" if _allocated(os.stat(root)) is not None else "logical"
    except OSError:
        volume_device, basis = None, "logical"
    # Every root's identity first: a walk skips a folder another place owns, and a root an earlier
    # place already claims (two places resolving to one folder) is walked once, by that place.
    claimed: Dict[Tuple[int, int], str] = {}
    resolved: List[Tuple[Dict[str, Any], List[Tuple[str, Tuple[str, ...]]], Set[Tuple[int, int]]]] = []
    for row in table:
        own: Set[Tuple[int, int]] = set()
        roots: List[Tuple[str, Tuple[str, ...]]] = []
        for base, parts in row["roots"]:
            identity = _identity(os.path.join(base, *parts))
            if identity is not None and identity in claimed:
                continue
            if identity is not None:
                claimed[identity] = row["id"]
                own.add(identity)
            roots.append((base, parts))
        resolved.append((row, roots, own))
    every = frozenset(claimed)
    done: List[StoragePart] = []
    totals = [0, 0]  # bytes and files of the finished places

    def report(place_id: str) -> Callable[[str, int, int], None]:
        def send(current: str, counted: int, files: int) -> None:
            if progress is not None:
                progress({"started_at": started_at, "place": place_id, "current": current,
                          "places_done": len(results), "places_total": len(table),
                          "bytes": totals[0] + counted, "files": totals[1] + files, "done": list(done)})
        return send

    results: List[StoragePlace] = []
    seen: Set[Tuple[int, int]] = set()  # linked files count once across the whole count
    for row, roots, own in resolved:
        walk = _Walk(every - own, seen=seen, progress=report(row["id"]), clock=clock, sleep=sleep)
        counted = _walk_roots(roots, walk, _NAMED_PARTS.get(row["id"], ()))
        same_volume = all(_same_device(os.path.join(base, *parts), volume_device) for base, parts in roots)
        place: StoragePlace = {"id": row["id"], "group": row["group"],
                               "roots": [os.path.join(base, *parts) for base, parts in roots],
                               "same_volume": same_volume, **counted}
        if row.get("undecided"):  # a root that could not be chosen: one unread item of unknown size
            place.update(present=True, bytes=None, unreadable=place["unreadable"] + 1, undecided=row["undecided"])
        results.append(place)
        if place["bytes"] is not None:
            done.append({"name": row["id"], "bytes": place["bytes"], "files": place["files"]})
        totals[0] += counted["bytes"] or 0
        totals[1] += counted["files"]
    return {
        "started_at": started_at,
        "finished_at": utc_now_iso(),
        "duration_seconds": round(max(0.0, clock() - began), 2),
        "io_priority": io_priority,
        "basis": basis,
        "bytes": totals[0],
        "logical_bytes": sum(place["logical_bytes"] for place in results),
        "files": totals[1],
        "unreadable": sum(place["unreadable"] for place in results),
        "volume": _volume(root),
        "places": results,
    }


def _same_device(path: str, device: Optional[int]) -> bool:
    try:
        return device is None or os.stat(path).st_dev == device
    except OSError:
        return True  # absent: nothing to be elsewhere


def size_paths(paths: List[str], *, progress: Optional[Callable[[Dict[str, Any]], None]] = None,
               each: Optional[Callable[[Dict[str, Any]], None]] = None,
               clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep) -> List[Dict[str, Any]]:
    """Size each directory in PATHS on its own (a link counted once inside one folder, never
    across folders, so an item's size is what that folder occupies). EACH receives every result
    as it is known; PROGRESS a ``{done, total, bytes}`` dict at most every 250 ms."""
    sizes: List[Dict[str, Any]] = []
    total_bytes = 0
    for index, path in enumerate(paths):
        def send(_current: str, counted: int, _files: int, index=index) -> None:
            if progress is not None:
                progress({"done": index, "total": len(paths), "bytes": total_bytes + counted})
        walk = _Walk(progress=send, clock=clock, sleep=sleep)
        counted = _walk_roots([(os.path.realpath(path), ())], walk)
        row = {"path": path, "bytes": counted["bytes"], "files": counted["files"], "unreadable": counted["unreadable"]}
        total_bytes += counted["bytes"] or 0
        sizes.append(row)
        if each is not None:
            each(row)
    return sizes


# ---- the saved file -------------------------------------------------------------------------

_REQUIRED = {"started_at": str, "finished_at": str, "bytes": int, "files": int, "unreadable": int, "places": list}


def load_saved(data_root: Any) -> Optional[Dict[str, Any]]:
    """The saved file when it is this schema and carries a finished count; else None ("no count
    yet", never zero): absent, unreadable, foreign or malformed alike."""
    raw = read_json_dict(pathlib.Path(data_root) / MEASUREMENT_FILE)
    if not raw or raw.get("_schema_version") != SCHEMA_VERSION:
        return None
    latest = raw.get("latest")
    if not isinstance(latest, dict) or any(not isinstance(latest.get(key), kind) or isinstance(latest.get(key), bool)
                                           for key, kind in _REQUIRED.items()):
        return None
    if not all(isinstance(place, dict) and isinstance(place.get("id"), str) for place in latest["places"]):
        return None
    previous = raw.get("previous")
    if not _is_summary(previous):
        previous = None  # a malformed summary loses growth, never the count or the reader
    return {"latest": latest, "previous": previous}


def _is_summary(value: Any) -> bool:
    """The shape ``_summary`` writes: the only ``previous`` ``with_growth`` and the mind's fact read."""
    if not isinstance(value, dict) or any(not isinstance(value.get(key), kind) or isinstance(value.get(key), bool)
                                          for key, kind in _REQUIRED.items() if key != "started_at"):
        return False
    return all(isinstance(place, dict) and isinstance(place.get("id"), str) for place in value["places"])


def _summary(latest: Dict[str, Any]) -> Dict[str, Any]:
    return {"finished_at": latest["finished_at"], "bytes": latest["bytes"], "files": latest["files"],
            "unreadable": latest["unreadable"],
            "places": [{"id": place["id"], "bytes": place.get("bytes"), "unreadable": place.get("unreadable", 0)}
                       for place in latest["places"]]}


def save_count(data_root: Any, latest: StorageCount) -> None:
    """Keep LATEST as the saved count and the count it replaces as ``previous`` (the last two
    finished counts; only a finished walk calls this, from the child)."""
    saved = load_saved(data_root)
    atomic_write_json(pathlib.Path(data_root) / MEASUREMENT_FILE, {
        "_schema_version": SCHEMA_VERSION, "latest": latest,
        "previous": _summary(saved["latest"]) if saved else None,
    })


def _growth(now: Any, then: Any) -> Optional[int]:
    return now - then if isinstance(now, int) and isinstance(then, int) else None


def with_growth(latest: Dict[str, Any], previous: Optional[Dict[str, Any]]) -> StorageCount:
    """LATEST with ``growth_bytes``/``growth_since`` against PREVIOUS on the total and on each
    place both counts hold, flagged when either count had unreadable items there."""
    count = {**latest, "places": [dict(place) for place in latest["places"]]}
    if not previous or not isinstance(previous.get("finished_at"), str):
        return count  # type: ignore[return-value]
    count["growth_bytes"] = _growth(latest.get("bytes"), previous.get("bytes"))
    count["growth_since"] = previous["finished_at"]
    if latest.get("unreadable") or previous.get("unreadable"):
        count["growth_flag"] = "unreadable"
    before = {place.get("id"): place for place in previous.get("places") or [] if isinstance(place, dict)}
    for place in count["places"]:
        then = before.get(place["id"])
        if then is None:
            continue
        place["growth_bytes"] = _growth(place.get("bytes"), then.get("bytes"))
        if place.get("unreadable") or then.get("unreadable"):
            place["growth_flag"] = "unreadable"
    return count  # type: ignore[return-value]


# ---- the server half ------------------------------------------------------------------------

class StorageCounter:
    """The server's count owner (module docstring): one child at a time, progress streamed from
    its stdout, the saved file read when its identity changes, Stop and the exit census."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stopped = threading.Event()
        self._cancel = False
        self._process = None
        self._thread: Optional[threading.Thread] = None
        self._counting: Optional[StorageCounting] = None
        self._error: Optional[StorageError] = None
        self._saved_key: Any = None
        self._saved: Optional[Dict[str, Any]] = None

    def saved(self, data_root: Any) -> Optional[Dict[str, Any]]:
        """The saved file, re-read only when its path or identity changed (one ``stat``)."""
        path = pathlib.Path(data_root) / MEASUREMENT_FILE
        try:
            info = path.stat()
            key = (str(path), info.st_mtime_ns, info.st_size)
        except OSError:
            key = (str(path), None, None)
        with self._lock:
            if key != self._saved_key:
                self._saved_key, self._saved = key, (load_saved(data_root) if key[1] is not None else None)
            return self._saved

    def projection(self, data_root: Any, *, count_token: Optional[str] = None,
                   free: Optional[Any] = None) -> StorageResponse:
        """Passive: the saved count with growth, the running count, the last error, fresh free space,
        the free-space tray (``free``: a ``storage_free.FreeSpace``, the module's own by default) and
        each place's removers; ``count_token`` equal to the latest ``finished_at`` omits the unchanged count."""
        from ouroboros import storage_free, storage_removers

        saved = self.saved(data_root)
        latest = with_growth(saved["latest"], saved["previous"]) if saved else None
        with self._lock:
            counting = dict(self._counting) if self._counting else None
            error = dict(self._error) if self._error else None
        response: StorageResponse = {  # type: ignore[typeddict-item]
            "counting": counting, "error": error, "volume": _volume(data_root),
            "retention_days": _retention_days(), "free": (free or storage_free.free).projection(data_root),
            "removers": storage_removers.place_removers(data_root, recent=True),  # 1 s polls share one engine read
        }
        if not (latest and count_token and count_token == latest["finished_at"]):
            response["count"] = latest
        return response

    def start(self, data_root: Any, repo_dir: Any) -> bool:
        """Start one count unless one runs; False means the caller joined the running one."""
        with self._lock:
            if self._stopped.is_set() or (self._thread is not None and self._thread.is_alive()):
                return False
            self._cancel = False
            self._error = None
            self._counting = {"started_at": utc_now_iso(), "place": "", "current": "", "places_done": 0,
                              "places_total": len(PLACES), "bytes": 0, "files": 0, "done": []}
            self._thread = threading.Thread(target=self._run, args=(pathlib.Path(data_root), pathlib.Path(repo_dir)),
                                            name="storage-count", daemon=True)
            try:
                self._thread.start()
            except RuntimeError:
                self._counting, self._thread = None, None
                self._error = {"at": utc_now_iso(), "message": "The count could not be started.", "stopped": False}
                return False
        return True

    def cancel(self) -> bool:
        """Kill the running child; False when nothing runs. The saved count stays."""
        with self._lock:
            if self._thread is None or not self._thread.is_alive():
                return False
            self._cancel = True
            process = self._process
        if process is not None:
            self._signal(process)
        return True

    def stop(self) -> None:
        self._stopped.set()
        self.cancel()

    def live_children(self) -> list:
        """Exit census after Stop: include an unfinished spawn until its owner returns."""
        process, thread = self._process, self._thread
        if process is not None and process.poll() is None:
            return [process]
        return [thread] if thread is not None and thread.is_alive() else []

    @staticmethod
    def _signal(process) -> None:
        with contextlib.suppress(OSError):
            process.kill()  # its own process group: the signal reaches only our child

    def _run(self, data_root: pathlib.Path, repo_dir: pathlib.Path) -> None:
        from ouroboros.startup_historical_audit import read_job_lines, spawn_maintenance_job

        started = time.monotonic()
        final: Dict[str, Any] = {}
        exit_code: Optional[int] = None
        failure = ""

        def on_line(raw: bytes) -> None:
            try:
                line = json.loads(raw)
            except ValueError:
                return
            if not isinstance(line, dict):
                return
            if line.get("event") == "progress":
                with self._lock:
                    if self._counting is not None:
                        self._counting = {**self._counting, **{key: line[key] for key in StorageCounting.__annotations__
                                                               if key in line and key != "started_at"}}
            elif "status" in line:
                final.update(line)

        try:
            process = spawn_maintenance_job(data_root, repo_dir, PURPOSE)
            with self._lock:
                self._process = process
            if self._cancel or self._stopped.is_set():
                self._signal(process)
            exit_code = read_job_lines(process, on_line, self._signal)
        except Exception as exc:
            failure = type(exc).__name__
            log.warning("Storage count could not run", exc_info=True)
        finally:
            with self._lock:
                self._process = None
        if final.get("status") == "finished":
            phase, error = "finished", None
        elif self._cancel or self._stopped.is_set():
            phase, error = "stopped", {"at": utc_now_iso(), "message": "Stopped before it finished.", "stopped": True}
        else:
            message = str(final.get("message") or "") if final.get("status") == "failed" else ""
            phase, error = "failed", {"at": utc_now_iso(), "stopped": False,
                                      "message": message or "The count ended without a result."}
        with self._lock:
            self._counting, self._error = None, error
        self._record(data_root, phase, exit_code, time.monotonic() - started, final, failure)

    @staticmethod
    def _record(data_root: pathlib.Path, phase: str, exit_code: Optional[int], seconds: float,
                final: Dict[str, Any], failure: str) -> None:
        from ouroboros.store import supervisor_row

        facts: Dict[str, Any] = {"phase": phase, "exit_code": exit_code, "duration_seconds": round(seconds, 2)}
        for key in ("bytes", "files", "unreadable", "io_priority"):
            if key in final:
                facts[key] = final[key]
        if failure:
            facts["exception_class"] = failure
        supervisor_row(data_root, "storage_count", facts)


def _retention_days() -> int:
    from ouroboros.retention import get_gc_retention_days

    return get_gc_retention_days()


count = StorageCounter()


# ---- the child half -------------------------------------------------------------------------

def utility_io(*, thread: bool = False) -> Optional[Callable[[], None]]:
    """macOS: the disk policy of this process, or with THREAD of the calling thread alone, becomes
    IOPOL_UTILITY and it never materializes dataless (cloud-evicted) files, so a mass read or rewrite
    never starves the Mac's foreground IO. Returns what puts back the two policies it replaced, None when
    nothing took; a no-op elsewhere. The count child keeps the process scope for its life; the
    ``record_compression`` job takes the thread scope around its own run and restores it after."""
    if sys.platform != "darwin":
        return None
    try:
        import ctypes

        # <sys/resource.h>: IOPOL_UTILITY is 4 (5 is IOPOL_STANDARD; a thread's 0 is "as its process").
        IOPOL_TYPE_DISK, IOPOL_TYPE_VFS_MATERIALIZE_DATALESS_FILES = 0, 3
        IOPOL_UTILITY, IOPOL_MATERIALIZE_DATALESS_FILES_OFF = 4, 1
        scope = 1 if thread else 0  # IOPOL_SCOPE_THREAD, IOPOL_SCOPE_PROCESS
        libc = ctypes.CDLL(None, use_errno=True)
        kinds = (IOPOL_TYPE_DISK, IOPOL_TYPE_VFS_MATERIALIZE_DATALESS_FILES)
        replaced = [(kind, libc.getiopolicy_np(kind, scope)) for kind in kinds]
        if libc.setiopolicy_np(IOPOL_TYPE_DISK, scope, IOPOL_UTILITY) != 0:
            return None
        libc.setiopolicy_np(IOPOL_TYPE_VFS_MATERIALIZE_DATALESS_FILES, scope, IOPOL_MATERIALIZE_DATALESS_FILES_OFF)
    except (OSError, AttributeError, ValueError):
        log.debug("Utility disk policy unchanged", exc_info=True)
        return None

    def restore() -> None:
        for kind, policy in replaced:
            if policy >= 0:
                libc.setiopolicy_np(kind, scope, policy)

    return restore


def _lower_io_priority() -> str:
    """Best-effort disk priority of this count child; the policy applied, ``unchanged`` when none
    took. Function-local platform calls under ``sys.platform`` guards (DEVELOPMENT 10)."""
    try:
        if sys.platform == "darwin":
            if utility_io() is not None:
                return "utility"
        elif sys.platform.startswith("linux"):
            import ctypes
            import platform

            number = {"x86_64": 251, "aarch64": 30}.get(platform.machine())
            if number is not None:
                libc = ctypes.CDLL(None, use_errno=True)
                # ioprio_set(IOPRIO_WHO_PROCESS, this process, IOPRIO_CLASS_BE << 13 | lowest level)
                if libc.syscall(number, 1, 0, (2 << 13) | 7) == 0:
                    return "best-effort-low"
        elif sys.platform == "win32":
            import ctypes

            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            if kernel32.SetPriorityClass(kernel32.GetCurrentProcess(), 0x00100000):  # PROCESS_MODE_BACKGROUND_BEGIN
                return "background"
    except (OSError, AttributeError, ValueError):
        log.debug("Storage count disk priority unchanged", exc_info=True)
    return "unchanged"


def _emit(line: Dict[str, Any]) -> None:
    print(json.dumps(line, separators=(",", ":")), flush=True)


def _sized_path(line: str) -> Optional[str]:
    """One ``--size-paths`` request line: a JSON string (ASCII on the pipe, so the platform's
    stdin code page cannot garble a path); anything else is ignored."""
    try:
        path = json.loads(line) if line.strip() else None
    except ValueError:
        return None
    return path if isinstance(path, str) and path else None


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--custody-session", default="")
    parser.add_argument("--size-paths", action="store_true",
                        help="size only the directories named one per line on stdin; save nothing")
    args = parser.parse_args(argv)
    root = pathlib.Path(args.data_root).resolve()
    os.environ["OUROBOROS_DATA_DIR"] = str(root)
    # Roots are bound before importing readers that capture config globals.
    from ouroboros.maintenance import _lower_priority
    from ouroboros.process_custody import adopt_session_id, start_parent_lifeline

    adopt_session_id(args.custody_session)
    start_parent_lifeline(label=PURPOSE)
    _lower_priority()
    io_priority = _lower_io_priority()
    try:
        if args.size_paths:
            paths = [path for path in map(_sized_path, sys.stdin) if path is not None]
            size_paths(paths, progress=lambda facts: _emit({"event": "progress", **facts}),
                       each=lambda row: _emit({"event": "size", **row}))
            _emit({"status": "finished", "io_priority": io_priority, "paths": len(paths)})
            return 0
        latest = measure(root, io_priority=io_priority, progress=lambda facts: _emit({"event": "progress", **facts}))
        try:
            save_count(root, latest)
        except OSError as exc:
            _emit({"status": "failed", "message": f"The count could not be saved: {type(exc).__name__}."})
            return 1
        _emit({"status": "finished", "finished_at": latest["finished_at"], "bytes": latest["bytes"],
               "files": latest["files"], "unreadable": latest["unreadable"],
               "duration_seconds": latest["duration_seconds"], "io_priority": io_priority})
        return 0
    except StorageUnavailable as exc:
        _emit({"status": "failed", "message": str(exc)})
    except Exception as exc:
        log.warning("Storage count failed", exc_info=True)
        _emit({"status": "failed", "message": f"The count failed: {type(exc).__name__}."})
    return 1


__all__ = [
    "ENGINE_DIR",
    "GROUPS",
    "MEASUREMENT_FILE",
    "PLACES",
    "StorageCount",
    "StorageCounter",
    "StorageCounting",
    "StorageError",
    "StoragePart",
    "StoragePlace",
    "StorageResponse",
    "StorageRest",
    "StorageUnavailable",
    "StorageVolume",
    "count",
    "load_saved",
    "measure",
    "place_roots",
    "save_count",
    "size_paths",
    "with_growth",
]


if __name__ == "__main__":
    sys.exit(main())

"""Dashboard -> Storage: the working folders deleted Projects left behind, as one count saw them.

Deleting a Project keeps its folder (``projects_registry``). A count learns here which folders those are and how they stand
against the Projects that remain: one registry read and one ``realpath`` and ``stat`` per registered folder (the deleted
Projects', and the remaining ones' only once a deleted one lists a folder), never a listing. These run
once, before the count's first place: a folder on a disk that does not answer them holds the whole count
there until Stop, which then keeps the previous count. Their sizes come from the
count's own walk (``storage.measure``): a ``FolderMark``, found by the folder's ``(device, inode)``, collects what the walk
attributes while it is inside that folder, so a folder inside a counted place costs no second listing and its size is a slice
of that place's. A folder no place holds is walked after every place (the owner's choice that Measure measures them too), and
its bytes never join the headline. One folder of several Projects is one row, a folder inside another marked folder is part of
that one (``within``), and a place's root inside a folder stays that place's (``holds``). A hard link counts where the walk met
it first, as everywhere in a count. Nothing here is deletion authority or a promise of space; the owner decides, elsewhere.
"""

from __future__ import annotations

import os
import stat
from typing import Any, Dict, List, Literal, Optional, Set, Tuple

try:  # Python 3.11+
    from typing import NotRequired, TypedDict  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - CI supports Python 3.10.
    from typing_extensions import NotRequired, TypedDict  # type: ignore[assignment]

_FOLDER_PARTS = 8  # a row's "Largest parts", as many as a place shows


class StorageFolderPart(TypedDict):
    """One depth-1 entry of a folder (files directly in it form the part ``(files)``)."""

    name: str
    bytes: int


class StorageDeletedProject(TypedDict):
    """A deleted Project whose folder this is."""

    name: str
    deleted_at: str  # when its deletion committed; "" on a record without it
    provenance: str  # the registry's word: attached | cloned | genesis | none; "" on a record without it


class StorageActiveProject(TypedDict):
    """A Project that remains (closed ones too) and touches this folder: its folder is this one (``same``), lies inside this
    one (``holds``), or holds this one (``inside``)."""

    name: str
    relation: Literal["same", "holds", "inside"]


class StorageDeletedFolder(TypedDict):
    """One folder, whatever number of deleted Projects it served. ``bytes`` is None (unknown, never zero) unless ``state`` is
    ``measured`` or ``partial`` (at least that much); ``not_measured``: this count did not reach it before it stopped."""

    path: str
    state: Literal["measured", "partial", "unreadable", "absent", "not_measured"]
    bytes: Optional[int]
    unreadable: int  # entries inside it that could not be read
    counted_in: Optional[str]  # the place whose walk attributed these bytes (they are in the total); None: outside the total
    same_volume: bool  # on the data folder's disk
    projects: List[StorageDeletedProject]
    active_projects: List[StorageActiveProject]
    parts: List[StorageFolderPart]
    rest: Dict[str, int]  # count and bytes of the parts beyond the largest eight
    reason: NotRequired[Literal["permission_denied", "not_a_folder", "read_error"]]
    within: NotRequired[str]  # the path of the listed folder whose bytes already include this one's
    holds: NotRequired[List[str]]  # ids of the places whose roots lie inside: their bytes are theirs, not in this size


class StorageDeletedProjects(TypedDict):
    """Dated by the finished count that holds it. ``unavailable``: the registry could not be read, so some folders may remain."""

    state: Literal["counted", "unavailable"]
    no_folder: int  # deleted Projects with no folder on record: never a row, never zero bytes
    folders: List[StorageDeletedFolder]


class FolderMark:
    """One folder through one count: what the registry says of it and what the walk attributed to it."""

    __slots__ = ("path", "identity", "state", "reason", "same_volume", "projects", "active", "reached", "listed", "place",
                 "within", "holds", "bytes", "unreadable", "parts")

    def __init__(self, path: str, identity: Optional[Tuple[int, int]], state: str, reason: str, same_volume: bool) -> None:
        self.path, self.identity, self.state, self.reason, self.same_volume = path, identity, state, reason, same_volume
        self.projects: List[StorageDeletedProject] = []
        self.active: List[StorageActiveProject] = []
        self.reached = self.listed = False
        self.place: Optional[str] = None
        self.within = ""
        self.holds: Set[str] = set()
        self.bytes = self.unreadable = 0
        self.parts: Dict[str, int] = {}

    def enter(self, place: Optional[str], outer: Optional["FolderMark"]) -> None:
        """The walk reached this folder, for PLACE (None: walked for itself) and inside the marked folder OUTER, if any."""
        self.reached, self.place, self.within = True, place, outer.path if outer else ""

    def add(self, part: str, size: int) -> None:
        self.bytes += size
        self.parts[part] = self.parts.get(part, 0) + size

    def row(self) -> StorageDeletedFolder:
        if self.listed:
            self.state = "partial" if self.unreadable else "measured"
        elif self.reached and self.state == "not_measured":  # reached and refused: what stands there now says why
            self.state, self.reason = _refused(self.path)
        ranked = sorted(self.parts.items(), key=lambda item: (-item[1], item[0]))
        row: StorageDeletedFolder = {
            "path": _text(self.path), "state": self.state, "bytes": self.bytes if self.listed else None,  # type: ignore[typeddict-item]
            "unreadable": self.unreadable, "counted_in": self.place, "same_volume": self.same_volume,
            "projects": sorted(self.projects, key=lambda project: project["deleted_at"]),
            "active_projects": sorted(self.active, key=lambda project: (project["relation"], project["name"])),
            "parts": [{"name": _text(name), "bytes": size} for name, size in ranked[:_FOLDER_PARTS]],
            "rest": {"count": len(ranked[_FOLDER_PARTS:]), "bytes": sum(size for _name, size in ranked[_FOLDER_PARTS:])},
        }
        if self.reason:
            row["reason"] = self.reason  # type: ignore[typeddict-item]
        if self.within:
            row["within"] = _text(self.within)
        if self.holds:
            row["holds"] = sorted(self.holds)
        return row


OpenMarks = Tuple[Tuple[FolderMark, str], ...]  # the marks a walked folder counts toward, each with the depth-1 part it is in there


def marks_below(open_marks: OpenMarks, name: str, own: Optional[FolderMark], place: Optional[str]) -> OpenMarks:
    """What the folder NAME inside OPEN_MARKS counts toward; OWN, its mark if it is registered, opens here for the walk of
    PLACE. Empty for nearly every folder of a count."""
    marks = tuple((mark, part or name) for mark, part in open_marks)
    if own is None or own.reached:  # reached already (a mount seen twice): its bytes are never attributed to it again
        return marks
    own.enter(place, marks[-1][0] if marks else None)
    return (*marks, (own, ""))


def _text(name: str) -> str:
    """A name a filesystem may hold in any bytes, in text the saved file can encode."""
    return name.encode("utf-8", "replace").decode("utf-8")


def _located(where: str) -> Tuple[str, str, str, Optional[Tuple[int, int]]]:
    """What stands at a registered folder now: ``(resolved path, state, reason, identity)``, the state ``not_measured`` when a
    folder does."""
    try:
        if not os.path.isabs(where):
            return where, "unreadable", "not_a_folder", None
        path = os.path.realpath(where)
    except ValueError:
        return where, "unreadable", "read_error", None
    try:
        info = os.stat(path)
    except (FileNotFoundError, NotADirectoryError):
        return path, "absent", "", None
    except PermissionError:
        return path, "unreadable", "permission_denied", None
    except OSError:
        return path, "unreadable", "read_error", None
    if not stat.S_ISDIR(info.st_mode):
        return path, "unreadable", "not_a_folder", None
    return path, "not_measured", "", (info.st_dev, info.st_ino) if info.st_ino else None


def _refused(path: str) -> Tuple[str, str]:
    """``(state, reason)`` of a folder the walk could not list."""
    _path, state, reason, _identity = _located(path)
    if state != "not_measured":
        return state, reason
    try:
        os.scandir(path).close()
    except FileNotFoundError:
        return "absent", ""
    except PermissionError:
        return "unreadable", "permission_denied"
    except OSError:
        pass
    return "unreadable", "read_error"


def _registry_rows(data_root: Any) -> List[Dict[str, Any]]:
    """Every readable row of the registry; a registry missing beside its commit witness is lost, never "no Projects"."""
    from ouroboros.project_admission import display_registry_snapshot
    from ouroboros.projects_registry import _registry_path, _registry_witness_path

    if _registry_witness_path(data_root).exists() and not _registry_path(data_root).exists():
        raise FileNotFoundError(str(_registry_path(data_root)))
    return display_registry_snapshot(data_root)["projects"]


def _name(row: Dict[str, Any], path: str) -> str:
    from ouroboros.project_facts import project_display_name

    return _text(project_display_name(row.get("name"), row.get("id")) or os.path.basename(path.rstrip(os.sep)) or str(row.get("id")))


def _above(path: str) -> List[str]:
    """Every folder above PATH."""
    above = []
    while os.path.dirname(path) != path:
        path = os.path.dirname(path)
        above.append(path)
    return above


def _link_active(rows: List[Dict[str, Any]], folders: Dict[Any, FolderMark]) -> None:
    """Each listed folder's remaining Projects: their folder is this one, lies inside it, or holds it (compared as the
    registry's own folder census compares them, ``normcase(realpath)``; one folder of two spellings also by identity)."""
    from ouroboros.projects_registry import PROJECT_ACTIVE

    listed = {os.path.normcase(mark.path): mark for mark in folders.values()}
    active: Dict[str, List[str]] = {}
    for row in rows:
        where = row.get("working_dir", "").strip()
        if row.get("lifecycle") != PROJECT_ACTIVE or not where:
            continue
        path, _state, _reason, identity = _located(where)
        key, name = os.path.normcase(path), _name(row, path)
        same = folders.get(identity) or listed.get(key)
        if same is not None:
            same.active.append({"name": name, "relation": "same"})
        active.setdefault(key, []).append(name)
        for parent in _above(key):
            if parent in listed:
                listed[parent].active.append({"name": name, "relation": "holds"})
    for key, mark in listed.items():
        for parent in _above(key):
            mark.active.extend({"name": name, "relation": "inside"} for name in active.get(parent, ()))


def read_folders(data_root: Any, device: Optional[int]) -> Tuple[str, int, List[FolderMark]]:
    """``(state, no_folder, one mark per folder)`` of the registry's deleted Projects; DEVICE is the data folder's disk."""
    from ouroboros.projects_registry import PROJECT_TOMBSTONED

    try:
        rows = _registry_rows(data_root)
    except (OSError, ValueError):
        return "unavailable", 0, []
    no_folder = 0
    folders: Dict[Any, FolderMark] = {}
    for row in rows:
        if row.get("lifecycle") != PROJECT_TOMBSTONED:
            continue
        where = row.get("working_dir", "").strip()
        if not where:
            no_folder += 1
            continue
        path, state, reason, identity = _located(where)
        same_volume = identity is None or device is None or identity[0] == device
        mark = folders.setdefault(identity or os.path.normcase(path), FolderMark(path, identity, state, reason, same_volume))
        mark.projects.append({"name": _name(row, path), "deleted_at": str(row.get("tombstoned_at") or ""),
                              "provenance": str(row.get("provenance") or "")})
    if folders:  # relations exist only for a listed folder: with none, no remaining Project's folder is looked up
        _link_active(rows, folders)
    return "counted", no_folder, list(folders.values())


def outside_order(marks: List[FolderMark]) -> List[FolderMark]:
    """The order the folders no place holds are walked in: the data folder's disk first, shorter paths first, and never a folder
    before an outside folder that holds it (a mount inside one): the outermost holder's disk decides for both, so the inner
    one is met on that walk and counted once. Paths compare as ``_link_active`` compares them."""
    by_path = {os.path.normcase(mark.path): mark for mark in marks}

    def disk(mark: FolderMark) -> bool:
        holders = [by_path[path] for path in _above(os.path.normcase(mark.path)) if path in by_path]
        return (holders[-1] if holders else mark).same_volume

    return sorted(marks, key=lambda mark: (not disk(mark), len(mark.path)))


def saved_folders(state: str, no_folder: int, marks: List[FolderMark]) -> StorageDeletedProjects:
    """What the count keeps: the folders as far as its walk has reached them."""
    return {"state": state, "no_folder": no_folder, "folders": [mark.row() for mark in marks]}  # type: ignore[typeddict-item]

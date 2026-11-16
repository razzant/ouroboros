"""Typed leaf: a Project's working folder as ``/api/projects/{project_id}/folder`` reads and changes it, and the
folder browser's listing (``GET /api/fs/dirs``) that the folder dialog and the New Project picker share.

Producers: ``ouroboros/project_folder.py``; the listing, ``gateway/projects.py::api_fs_dirs``. Browser twins:
``web/modules/project_folder_types.js``; the listing's, ``web/modules/api_types.js``. ``gateway/contracts.py``
re-exports the listing.
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

try:
    from typing import NotRequired, TypedDict  # type: ignore[attr-defined]
except ImportError:
    from typing_extensions import NotRequired, TypedDict  # type: ignore[assignment]


class ProjectFolderWorkItem(TypedDict):
    task_id: str
    name: str
    system_repo: bool  # its recorded intent is Ouroboros's own repository: a chosen folder would take it from there


class ProjectFolderWork(TypedDict):
    """The room's queued roots by what a folder change means for them."""

    observed: bool  # false: the queue or the bindings could not be read, so nothing is counted
    kept: int  # admitted work: keeps the resources it was admitted with, through Resume and Continue
    moved: List[ProjectFolderWorkItem]  # moved here, not restarted since: takes the room's folder at its next start
    moved_total: int  # every such work; ``moved`` names the first 20


class ProjectFolderView(TypedDict):
    """One room's folder choice and what one look at it found. ``actions`` is empty for a folder in place."""

    project_id: str
    name: str
    chat_id: int
    folder: str  # the registered address, "" when none
    state: Literal["usable", "missing", "unusable", "unset", "none"]  # unset: no choice made; none: the recorded No folder
    reason: str  # why a set folder is missing or unusable, else ""
    no_folder: Optional[Dict[str, str]]  # {at, by} of the recorded choice
    provenance: str
    actions: List[Literal["clear", "choose"]]
    expected: str  # sent back with a change; a choice that moved meanwhile answers 409
    work: ProjectFolderWork


class ProjectFolderRequest(TypedDict):
    """``POST``: ``clear`` records No folder; ``choose`` attaches the existing folder ``path``."""

    action: Literal["clear", "choose"]
    expected: str
    path: NotRequired[str]


class ProjectFolderReceipt(TypedDict):
    """The change as appended to ``logs/events.jsonl``; ``recorded`` false when that append failed."""

    ts: str
    type: str
    project_id: str
    actor: str  # owner or ouroboros:<task id>
    action: Literal["clear", "choose"]
    old: Dict[str, Any]  # working_dir, no_folder, state, reason
    new: Dict[str, Any]  # working_dir, no_folder
    recorded: bool
    note: NotRequired[str]


class ProjectFolderResponse(TypedDict):
    folder: Optional[ProjectFolderView]  # null only beside a receipt: changed, the look after it failed (folder_error)
    receipt: NotRequired[ProjectFolderReceipt]
    folder_error: NotRequired[str]  # why that look failed; the change stands: read the folder again, never re-send


class FsDirsEntry(TypedDict):
    name: str
    path: str
    is_git: bool


class FsDirsResponse(TypedDict):
    """GET /api/fs/dirs — owner-facing directory browser for the New Project attach
    picker (server-side; works in web/Docker). Directories only, confined to the
    home tree, never file contents. ``truncated`` is True when the directory holds
    more children than the 500-entry cap (no silent truncation)."""

    path: str
    parent: str
    home: str
    dirs: List[FsDirsEntry]
    truncated: bool


__all__ = ["FsDirsEntry", "FsDirsResponse", "ProjectFolderReceipt", "ProjectFolderRequest", "ProjectFolderResponse",
           "ProjectFolderView", "ProjectFolderWork", "ProjectFolderWorkItem"]

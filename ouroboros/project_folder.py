"""A Project's working folder as its owner and its mind read and change it: one view, one change.

The registry row is the only authority: ``working_dir`` names the folder a room's new work starts
in, ``no_folder`` the recorded choice to keep none (``project_sources.apply_folder``). ``folder_view``
reads one room's choice together with what one look at that address found; ``change_folder`` changes
it through the registry's writer, compare-and-set on the view's ``expected``, and appends the receipt
(actor, old, new) to ``logs/events.jsonl``. The Project menu's "Working folder…"
(``gateway/projects.py``) calls these two. The mind's ``manage_project_folder`` reads through
``view_outcome`` (the view, or the ``unknown_project`` refusal ``change_folder`` also gives) and
changes through ``change_folder``.

The door is narrow on purpose. It shows any folder and changes one only where the room has no usable
default: a folder that is missing or unusable, no choice made yet, or the recorded No folder. A
folder that is in place stays. It attaches an existing durable folder or records No folder; it never
creates, clones, copies, restores or deletes one, and it retargets no admitted work: a queue row
keeps the resources it was admitted with, through Resume and Continue. The one exception is the
placement contract itself (``project_placement.readmit_placement``): work moved into the room that
has not restarted since takes the room's default at its next start, even if it already ran
elsewhere, so the view names that work (``work.moved``).

Only the addressed room is inspected, on request: listing Projects never stats a folder.
"""
from __future__ import annotations

import hashlib
import json
import logging
import pathlib
from typing import Any, Optional

log = logging.getLogger(__name__)

# What one look found: the folder is in place; its address is absent; something is there that cannot
# serve (not a directory, a bare or non-root Git tree, Ouroboros's own repository or data, a copy the
# host removes, an address the filesystem refuses to examine); the row names no folder and no choice;
# the recorded choice to keep none.
USABLE, MISSING, UNUSABLE, UNSET, NONE = "usable", "missing", "unusable", "unset", "none"
CLEAR, CHOOSE = "clear", "choose"
_OFFERED = {USABLE: (), MISSING: (CLEAR, CHOOSE), UNUSABLE: (CLEAR, CHOOSE), UNSET: (CLEAR, CHOOSE), NONE: (CHOOSE,)}
_MOVED_SHOWN = 20  # names listed; ``moved_total`` counts every one


def _expected(row: dict) -> str:
    """The choice a caller saw, as one opaque word: the routing identity admission compares, with the
    folder and the recorded No folder. Any change of either starts a new routing generation."""
    seen = [row.get(key) for key in ("id", "chat_id", "lifecycle", "routing_generation", "working_dir",
                                     "routing_incarnation", "created_at", "no_folder")]
    return hashlib.sha256(json.dumps(seen, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]


def _observed(row: dict, *, repo_dir: Any, drive_root: Any) -> tuple[str, str]:
    """``(state, reason)`` of the row's folder at this moment, by the checks admission itself applies. A look the
    filesystem refuses (no search permission on the way, a symlink loop) found a folder that cannot serve: unusable,
    with that error as its reason, so the view and its repair stay open."""
    folder = str(row.get("working_dir") or "").strip()
    if not folder:
        return (NONE, "") if "no_folder" in row else (UNSET, "")
    from ouroboros.project_sources import temporary_copy_refusal
    from ouroboros.workspace_admission import WorkspaceRootError, unavailable_workspace_address, validate_workspace_root

    try:
        temporary = temporary_copy_refusal(folder, drive_root=drive_root)
        try:
            validate_workspace_root(folder, system_repo_dir=repo_dir, drive_root=drive_root)
        except WorkspaceRootError as exc:
            absent = unavailable_workspace_address(folder, system_repo_dir=repo_dir, drive_root=drive_root)
            return (MISSING if absent else UNUSABLE), "; ".join(part for part in (str(exc), temporary) if part)
    except (OSError, RuntimeError) as exc:  # RuntimeError: pathlib's symlink loop before Python 3.13
        return UNUSABLE, f"{folder} cannot be examined: {type(exc).__name__}: {exc}"
    return (UNUSABLE, temporary) if temporary else (USABLE, "")


def _work(drive_root: Any, project_id: str) -> dict:
    """The room's work in the queue's last publication, by what a folder change means for it.

    ``kept`` roots keep the resources they were admitted with. ``moved`` roots carry a recorded move
    into this room that no start has applied yet and keep no resource of their own: readmission gives
    them the room's folder as it then stands, also to work whose recorded intent is Ouroboros's own
    repository (``system_repo``), which a chosen folder would take away from it. Helpers follow their
    roots and are not counted. ``observed`` false: the queue or the bindings could not be read, so
    nothing is counted."""
    from ouroboros.project_placement import keeps_own_resource, pending_placement
    from ouroboros.projects_registry import _load_bindings, task_presentation_name
    from ouroboros.queue_observation import load_queue_document

    kept, moved = 0, []
    unknown = {"observed": False, "kept": 0, "moved": [], "moved_total": 0}
    try:
        document = load_queue_document(drive_root)
        if document.get("_snapshot_missing") or document.get("_snapshot_invalid"):
            return unknown  # no publication to read is not "no work"
        bindings = _load_bindings(drive_root, strict=True)["bindings"]
        for lane in ("running", "pending"):
            for item in document.get(lane) or []:
                task = item.get("task") if isinstance(item, dict) else None
                if not isinstance(task, dict) or task.get("delegation_role") == "subagent":
                    continue
                task_id = str(task.get("id") or item.get("id") or "")
                task = task if task.get("id") else {**task, "id": task_id}
                binding = bindings.get(task_id)
                room = binding.get("project_id") if isinstance(binding, dict) else task.get("project_id")
                if str(room or "") != project_id:
                    continue
                if (pending_placement(drive_root, task, bindings_snapshot=bindings)
                        and not keeps_own_resource(drive_root, task)[0]):
                    intent = (task.get("metadata") or {}).get("resource_intent") or {}
                    moved.append({"task_id": task_id, "name": task_presentation_name(task),
                                  "system_repo": intent.get("kind") == "system_repo"})
                else:
                    kept += 1
    except Exception:
        log.warning("Project folder view: the room's queued work could not be read", exc_info=True)
        return unknown
    return {"observed": True, "kept": kept, "moved": moved[:_MOVED_SHOWN], "moved_total": len(moved)}


def _active_row(drive_root: Any, project_id: Any) -> Optional[dict]:
    from ouroboros.projects_registry import PROJECT_ACTIVE, get_reserved_project

    row = get_reserved_project(drive_root, str(project_id or ""), strict=True)
    return row if row and row.get("lifecycle") == PROJECT_ACTIVE else None


def folder_view(drive_root: Any, project_id: Any, *, repo_dir: Any, row: Optional[dict] = None) -> Optional[dict]:
    """One room's folder choice and what was observed of it now; ``None`` for no active Project.

    ``actions`` are the changes this state offers (none for a folder in place); ``expected`` is the
    word ``change_folder`` compares. Reads one registry row, one folder and the queue's publication."""
    row = _active_row(drive_root, project_id) if row is None else row
    if row is None:
        return None
    state, reason = _observed(row, repo_dir=repo_dir, drive_root=drive_root)
    return {
        "project_id": row["id"], "name": str(row.get("name") or row["id"]), "chat_id": row.get("chat_id"),
        "folder": str(row.get("working_dir") or "").strip(), "state": state, "reason": reason,
        "no_folder": dict(row["no_folder"]) if isinstance(row.get("no_folder"), dict) else None,
        "provenance": str(row.get("provenance") or ""),
        "actions": list(_OFFERED[state]), "expected": _expected(row),
        "work": _work(drive_root, row["id"]),
    }


def effects(view: dict, action: str, folder: str = "") -> list[str]:
    """What ``action`` changes and leaves, in plain sentences for a mind (the dialog words the same facts)."""
    work = view.get("work") or {}
    target = folder or "the chosen folder"
    lines = [
        ("New work in this Project starts in its own temporary area, and nothing creates a folder for it "
         "until one is chosen; work on Ouroboros's own repository stays there."
         if action == CLEAR else f"New work in this Project starts in {target}."),
        "No file, history or result is deleted or moved; only the Project's default changes.",
    ]
    if not work.get("observed"):
        lines.append("The room's queued work could not be read: which work keeps its folder is unknown.")
        return lines
    if work.get("kept"):
        lines.append(f"{work['kept']} admitted work(s) keep the folder they were admitted with, also after Resume or "
                     "Continue: this does not move them, and work admitted with a folder that is gone stays as it is.")
    listed = work.get("moved") or []
    for body in (True, False):
        names = ", ".join(str(item.get("name") or item.get("task_id")) for item in listed if bool(item.get("system_repo")) is body)
        if not names:
            continue
        lead = f"Moved into this Project and not restarted since: {names}. At its next start, even though it already ran, "
        if body:
            lines.append(lead + ("it stays on Ouroboros's own repository." if action == CLEAR else
                                 f"it would work in {target} instead of Ouroboros's own repository; clear keeps it there."))
        else:
            lines.append(lead + ("it gets no folder from this Project and works in its own temporary area."
                                 if action == CLEAR else f"it works in {target}."))
    more = int(work.get("moved_total") or 0) - len(listed)
    if more > 0:
        lines.append(f"{more} more moved work(s) are not listed; the same holds for them.")
    return lines


def _refused(reason: str, detail: str, view: Optional[dict]) -> dict:
    return {"status": "refused", "reason": reason, "detail": detail, "view": view}


def view_outcome(drive_root: Any, project_id: Any, *, repo_dir: Any) -> dict:
    """The view of an active Project, else the ``unknown_project`` refusal that ``change_folder`` gives."""
    view = folder_view(drive_root, project_id, repo_dir=repo_dir)
    if view is None:
        return _refused("unknown_project", "No active Project has this id.", None)
    return {"status": "view", "view": view}


def change_folder(drive_root: Any, project_id: Any, *, action: str, expected: str, actor: str, repo_dir: Any,
                  path: str = "", note: str = "") -> dict:
    """Record No folder (``clear``) or attach the existing durable folder ``path`` (``choose``).

    Answers ``{status: "changed", receipt, view}`` (``view`` None and ``view_error`` when the look after
    the change failed: the change stands, read it again) or ``{status: "refused", reason, detail, view}``
    with the view as it stands: ``unknown_project``; ``stale`` (``expected`` is not the current
    choice: read it again); ``not_offered`` (this state does not offer the action); ``invalid_folder``;
    ``changed`` (the row moved while this wrote). Nothing is written on a refusal. The folder is
    validated as an attach is: it exists, is no temporary host copy, and keeps Ouroboros's own
    repository and data apart; a folder the filesystem will not let it examine is invalid too.
    ``actor`` is ``owner`` or ``ouroboros:<task id>``."""
    from ouroboros.project_admission import ProjectAdmissionError, project_admission_basis
    from ouroboros.project_sources import TemporaryCopyRefused, validate_attach_path
    from ouroboros.projects_registry import update_project
    from ouroboros.utils import append_jsonl, utc_now_iso
    from ouroboros.workspace_admission import WorkspaceRootError, validate_workspace_root

    row = _active_row(drive_root, project_id)
    if row is None:
        return _refused("unknown_project", "No active Project has this id.", None)
    view = folder_view(drive_root, row["id"], repo_dir=repo_dir, row=row)
    if not isinstance(expected, str) or expected != view["expected"]:
        return _refused("stale", "The Project's folder choice changed since it was read.", view)
    if action not in view["actions"]:
        offered = " or ".join(view["actions"]) or "no change"
        return _refused("not_offered", f"A folder that is {view['state']} offers {offered} here.", view)
    now = utc_now_iso()
    if action == CHOOSE:
        try:  # a folder one may not look into (no search permission, a symlink loop) cannot serve: a refusal
            resolved, error = validate_attach_path(path, system_repo_dir=repo_dir, drive_root=drive_root)
            if not error:
                try:
                    resolved = validate_workspace_root(resolved, system_repo_dir=repo_dir, drive_root=drive_root)
                except WorkspaceRootError as exc:
                    error = str(exc)
        except (OSError, RuntimeError) as exc:  # RuntimeError: pathlib's symlink loop before Python 3.13
            error = f"{path} cannot be examined: {type(exc).__name__}: {exc}"
        if error:
            return _refused("invalid_folder", error, view)
        updates = {"working_dir": str(resolved), "no_folder": None, "provenance": "attached", "clone_url": "",
                   "trusted_at": now}
    else:
        updates = {"no_folder": {"at": now, "by": actor}}
    try:
        saved = update_project(drive_root, row["id"], admission_basis=project_admission_basis(row["id"], row), **updates)
    except ProjectAdmissionError as exc:
        return _refused("changed", str(exc), folder_view(drive_root, row["id"], repo_dir=repo_dir))
    except TemporaryCopyRefused as exc:  # the writer's own durable-folder check has the last word
        return _refused("invalid_folder", str(exc), view)
    if saved is None:
        return _refused("unknown_project", "The Project is no longer active.", None)
    receipt = {
        "ts": now, "type": "project_folder_changed", "project_id": row["id"], "actor": actor, "action": action,
        "old": {"working_dir": view["folder"], "no_folder": view["no_folder"], "state": view["state"],
                "reason": view["reason"]},
        "new": {"working_dir": str(saved.get("working_dir") or ""), "no_folder": saved.get("no_folder")},
        **({"note": str(note)} if str(note or "").strip() else {}),
    }
    try:
        receipt["recorded"] = bool(append_jsonl(pathlib.Path(drive_root) / "logs" / "events.jsonl", receipt))
    except Exception:
        log.warning("Project folder changed; its receipt could not be appended", exc_info=True)
        receipt["recorded"] = False
    try:  # the change stands whatever this look finds: a failed look leaves the view unknown, never the change
        return {"status": "changed", "receipt": receipt,
                "view": folder_view(drive_root, row["id"], repo_dir=repo_dir, row=saved)}
    except Exception as exc:
        log.warning("Project folder changed; the view after it could not be read", exc_info=True)
        return {"status": "changed", "receipt": receipt, "view": None, "view_error": f"{type(exc).__name__}: {exc}"}


__all__ = ["CHOOSE", "CLEAR", "change_folder", "effects", "folder_view"]

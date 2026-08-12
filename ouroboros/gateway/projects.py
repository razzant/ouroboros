"""Projects gateway handlers (multi-project, v6.32.0).

Thin transport over ``ouroboros.projects_registry`` — list/create plus the
per-project chat id the UI needs to open a project thread. No business logic
here (Gateway Boundary rule).
"""

from __future__ import annotations

import logging
from typing import Any

from starlette.requests import Request
from starlette.responses import JSONResponse

from ouroboros.gateway._helpers import json_exception, request_drive_root, request_repo_dir

log = logging.getLogger(__name__)

# Project name auto-derived from the task objective is capped here so the
# sidebar label stays readable; the live card keeps showing full progress.
_MAX_DERIVED_NAME = 60

def _task_from_live_queue(drive_root: object, task_id: str) -> dict:
    """The task dict of a still-RUNNING/PENDING task from the queue snapshot.

    A main-chat task's task_result carries its fields only once it is written (a
    plain chat task writes them at finish). But the owner converts a card while
    the task is IN-PROGRESS, so load_task_result can miss it and the name falls
    back to the bare id (observed live: task-ae349c73). The queue snapshot
    persists every PENDING/RUNNING task (title/objective/description) at
    assignment, so it is the reliable in-flight source. Never raises."""
    try:
        import json
        import pathlib

        snap = pathlib.Path(str(drive_root)) / "state" / "queue_snapshot.json"
        if not snap.exists():
            return {}
        data = json.loads(snap.read_text(encoding="utf-8"))
        for bucket in ("running", "pending"):
            for row in (data.get(bucket) or []):
                if not isinstance(row, dict):
                    continue
                task = row.get("task") if isinstance(row.get("task"), dict) else {}
                if str(task.get("id") or row.get("id") or "") == str(task_id):
                    return task
    except Exception:
        log.debug("_task_from_live_queue failed", exc_info=True)
    return {}


def _owner_request_text(drive_root: object, task_id: str, hint: str = "") -> str:
    """The owner's ORIGINAL request for a task, UNtruncated (unlike the 60-char
    project name). Preference: persisted/live ``objective`` (what the owner asked)
    then ``description`` then ``title``; finally the frontend ``objective_hint``
    (the owner's last main-chat request, for an in-progress DIRECT conversion with
    no server-side record yet). Used to identify the canonical owner row projected
    into the Project lens; the row itself is never copied. Never raises."""
    try:
        from ouroboros.task_results import load_task_result

        result = load_task_result(drive_root, task_id) or {}
    except Exception:
        log.debug("_owner_request_text: load_task_result failed", exc_info=True)
        result = {}
    live = _task_from_live_queue(drive_root, task_id)
    for field in ("objective", "description", "title"):
        for src in (result, live):
            value = str((src or {}).get(field) or "").strip()
            if value:
                return value
    return " ".join(str(hint or "").split())


def _owner_task_origin(drive_root: object, task_id: str) -> dict:
    """The typed binding origin for a post-hoc conversion of ``task_id``.

    Reads the ingress-captured ``origin_message_ref``/``origin_message_text``
    from the persisted task result or the live queue record (identity by value —
    never re-derived from content). A pre-v6.73.0 task without a captured origin
    converts with the typed ``post_hoc_unresolved`` reason: its start message is
    honestly not projectable, never silently empty."""
    sources = []
    try:
        # Freshest first: the authoritative IN-MEMORY queue (the gateway runs in
        # the supervisor's process), so a conversion clicked right after enqueue
        # — before any snapshot/task_result persistence — still finds the origin.
        import supervisor.queue as queue_mod
        from supervisor.queue import _queue_lock

        tid = str(task_id or "")
        with _queue_lock:
            for pending in queue_mod.PENDING:
                if isinstance(pending, dict) and str(pending.get("id") or "") == tid:
                    sources.append(dict(pending))
            running_meta = queue_mod.RUNNING.get(tid)
            if isinstance(running_meta, dict) and isinstance(running_meta.get("task"), dict):
                sources.append(dict(running_meta["task"]))
    except Exception:
        log.debug("_owner_task_origin in-memory queue lookup failed", exc_info=True)
    try:
        # Child-merging reader (scope r3 advisory): a forked/workspace ROOT's
        # running record lives on its CHILD drive; the effective-status SSOT
        # merges it so a post-hoc conversion of a terminal forked root still
        # finds the captured origin.
        from ouroboros.task_status import load_effective_task_result

        sources.append(load_effective_task_result(drive_root, task_id) or {})
        sources.append(_task_from_live_queue(drive_root, task_id) or {})
    except Exception:
        log.debug("_owner_task_origin lookup failed", exc_info=True)
    for source in sources:
        ref = source.get("origin_message_ref")
        if isinstance(ref, dict) and ref:
            text = source.get("origin_message_text")
            if not (isinstance(text, str) and text.strip()):
                # A malformed record (ref without its cross-thread text copy)
                # degrades to the typed absence — never an unhandled 500.
                continue
            return {"ref": dict(ref), "text": text}
    return {"absent": "post_hoc_unresolved"}


def _derive_project_name(drive_root: object, task_id: str) -> str:
    """Best-effort, NO-extra-request project name for a "turn into project" card.

    Names the project with zero human input and zero extra LLM call (owner P1).
    Preference order: the model-coined short ``title`` (set at card creation),
    then the task ``objective`` (the owner's original request), then
    ``description`` — each looked up first in the persisted task_result and then
    in the live queue snapshot (for an in-progress conversion). Finally an empty
    string so the caller supplies a generic id fallback. Never raises."""
    try:
        from ouroboros.task_results import load_task_result

        result = load_task_result(drive_root, task_id) or {}
    except Exception:
        log.debug("_derive_project_name: load_task_result failed", exc_info=True)
        result = {}
    live = _task_from_live_queue(drive_root, task_id)
    raw = ""
    for field in ("title", "objective", "description"):
        for src in (result, live):
            value = str((src or {}).get(field) or "").strip()
            if value:
                raw = value
                break
        if raw:
            break
    cleaned = " ".join(raw.split())
    if len(cleaned) > _MAX_DERIVED_NAME:
        cleaned = cleaned[: _MAX_DERIVED_NAME - 1].rstrip() + "…"
    return cleaned


def _task_workspace_root(drive_root: object, task_id: str) -> str:
    """The folder a task is (or was) working in — persisted result first, then the
    live queue snapshot for a conversion clicked mid-flight. Same two sources the
    name derivation reads, for the same reason: the owner converts a card while the
    task is still running, when only the snapshot knows. Never raises."""
    try:
        from ouroboros.task_results import load_task_result

        result = load_task_result(drive_root, task_id) or {}
    except Exception:
        log.debug("_task_workspace_root: load_task_result failed", exc_info=True)
        result = {}
    for src in (result, _task_from_live_queue(drive_root, task_id)):
        value = str((src or {}).get("workspace_root") or "").strip()
        if value:
            return value
    return ""


def _preset_suggested_name(drive_root: object, task_id: str) -> str:
    """The LLM title the proactive card namer already coined for this task (Cluster B),
    read from the persisted result then the live queue. Reused by turn-into-project so
    the conversion needs no extra LLM call. Empty when the namer has not run yet (a
    convert click within the first ~second). Never raises."""
    try:
        from ouroboros.task_results import load_task_result

        result = load_task_result(drive_root, task_id) or {}
    except Exception:
        log.debug("_preset_suggested_name: load_task_result failed", exc_info=True)
        result = {}
    live = _task_from_live_queue(drive_root, task_id)
    for src in (result, live):
        value = str((src or {}).get("suggested_name") or "").strip()
        if value:
            return value
    return ""


# Human labels for the skill-lifecycle job kinds that ``skill_lifecycle_queue.
# _chat_task_id`` encodes into a synthetic task id (skill_lifecycle_<kind>_<target>_<job>).
_SKILL_LIFECYCLE_KINDS = {
    "install": "Install skill",
    "review": "Review skill",
    "enable": "Enable skill",
    "disable": "Disable skill",
    "remove": "Remove skill",
    "update": "Update skill",
    "dependency": "Skill dependencies",
    "dependencies": "Skill dependencies",
}


def _skill_name_from_task(drive_root: object, task_id: str) -> str:
    """An explicit skill name carried by a skill/system task (``skill`` /
    ``metadata.skill`` / ``target``), persisted-result first then live queue.
    Empty if none. Never raises."""
    try:
        from ouroboros.task_results import load_task_result

        result = load_task_result(drive_root, task_id) or {}
    except Exception:
        result = {}
    live = _task_from_live_queue(drive_root, task_id)
    for src in (result, live):
        if not isinstance(src, dict):
            continue
        meta = src.get("metadata") if isinstance(src.get("metadata"), dict) else {}
        for value in (src.get("skill"), meta.get("skill"), src.get("target")):
            name = str(value or "").strip()
            if name:
                return name
    return ""


def _cap_name(name: str) -> str:
    name = " ".join(str(name or "").split())
    if len(name) > _MAX_DERIVED_NAME:
        return name[: _MAX_DERIVED_NAME - 1].rstrip() + "…"
    return name


def _system_task_display_name(drive_root: object, task_id: str) -> str:
    """A human project name for a NON-human (skill/system) task that carries no
    owner request text — so "turn into project" never dead-ends at the neutral
    "New project". Source order: an explicit ``skill`` field, then the structural
    ``skill_lifecycle_<kind>_<target>_<job>`` task-id form coined by
    ``skill_lifecycle_queue._chat_task_id``. NOT a semantic gate (P5): it reads an
    explicit field and a known structural id shape, never the objective text.
    Empty when the task is not a recognized system task. Never raises."""
    tid = str(task_id or "")
    explicit_skill = _skill_name_from_task(drive_root, tid)
    if tid.startswith("skill_lifecycle_"):
        parts = tid[len("skill_lifecycle_"):].split("_")
        kind = parts[0] if parts else ""
        kind_label = _SKILL_LIFECYCLE_KINDS.get(kind, ("Skill " + kind).strip() or "Skill task")
        # target = explicit skill field, else the id segments between kind and the
        # trailing sanitized job-id segment. Best-effort; the name is cosmetic.
        target = explicit_skill
        if not target and len(parts) >= 3:
            target = "_".join(parts[1:-1]).strip("_")
        elif not target and len(parts) == 2:
            target = parts[1].strip("_")
        target = " ".join(str(target or "").split())
        return _cap_name(f"{kind_label}: {target}" if target else kind_label)
    if explicit_skill:
        return _cap_name(f"Skill: {explicit_skill}")
    return ""


def _emit_naming_reason(drive_root: object, task_id: str, name: str, reason: str) -> None:
    """Durable structured telemetry for HOW a project was named (which fallback
    path fired) so a future "New project" regression is visible in events.jsonl
    instead of silent (north star: transparency). Best-effort; never raises."""
    try:
        import pathlib

        from ouroboros.utils import append_jsonl, utc_now_iso

        append_jsonl(
            pathlib.Path(str(drive_root)) / "logs" / "events.jsonl",
            {
                "ts": utc_now_iso(),
                "type": "project_named",
                "task_id": str(task_id),
                "name": str(name),
                "reason": str(reason),
            },
        )
    except Exception:
        log.debug("_emit_naming_reason failed", exc_info=True)


async def api_projects_list(request: Request) -> JSONResponse:
    """GET /api/projects — the sidebar's list, optionally WITH archived threads.

    ``?include_archived=1`` is the only way an archived thread ever reaches a
    surface, and it exists because without it archiving was a ONE-WAY trip:
    ``projects_summary`` is the only projection that lists threads, so a thread it
    filtered out could never be shown, which made ``POST …/restore`` and the
    ``restore`` row in the thread menu unreachable by construction (T3R-8). The
    default is unchanged, so the sidebar keeps hiding them.
    """
    try:
        from ouroboros.gateway.state import live_thread_chat_ids
        from ouroboros.projects_registry import (
            projects_summary,
        )

        drive_root = request_drive_root(request)
        include_archived = str(
            request.query_params.get("include_archived") or ""
        ).strip().lower() in {"1", "true", "yes", "on"}
        # Same visibility rule as /api/state unless the caller ASKS otherwise:
        # archived threads are hidden unless a task is still live in them (X10).
        # Two summaries disagreeing about which threads exist would be worse than
        # either answer alone — so the difference is REQUESTED, never incidental,
        # and the answer echoes which list this is.
        return JSONResponse({
            "projects": projects_summary(
                drive_root, limit=200, live_chat_ids=live_thread_chat_ids(),
                include_archived=include_archived,
            ),
            "include_archived": include_archived,
        })
    except Exception as exc:
        return json_exception(exc)


async def api_projects_create(request: Request) -> JSONResponse:
    """POST /api/projects — create a project from one of FOUR sources (v6.59.0):

    - ``path=``       attach an existing owner folder (validated on the RESOLVED
                      realpath; optional ``init_git`` makes an attach-snapshot
                      commit — NEVER auto-init without the flag);
    - ``git_url=``    server-side clone into the durable projects root (atomic
                      tmp→rename, non-interactive, typed ``auth_required``);
    - ``with_workspace`` provision a fresh genesis folder (pre-v6.59 behavior);
    - none of these   a file-less project (research/chat-only).

    ``provenance`` (attached|cloned|genesis|none) + ``clone_url`` are recorded as
    historical facts; ``trusted_at`` is stamped automatically for attach/clone
    (the notification trust model — attaching IS the owner's explicit grant).
    """
    try:
        import asyncio

        from ouroboros.project_facts import (
            explicit_project_id_ok,
            project_id_from_display_name,
            sanitize_project_id,
        )
        from ouroboros.projects_registry import (
            PROJECT_NAME_MAX,
            create_project,
            ensure_project_workspace,
            set_working_dir_if_absent,
            update_project,
        )
        from ouroboros.utils import utc_now_iso

        body = await request.json()
        if not isinstance(body, dict):
            return JSONResponse({"error": "body must be a JSON object"}, status_code=400)
        name = str(body.get("name") or "").strip()
        if len(name) > PROJECT_NAME_MAX:
            return JSONResponse(
                {"error": f"name must be <= {PROJECT_NAME_MAX} characters"},
                status_code=400,
            )
        raw_id = str(body.get("id") or body.get("project_id") or "").strip()
        if raw_id and not explicit_project_id_ok(raw_id):
            return JSONResponse(
                {"error": f"id {raw_id!r} is not filesystem-clean (lowercase alphanumeric/_/-/., <=64 chars)"},
                status_code=400,
            )
        if not raw_id:
            # Name-only creation (the New Project dialog): derive a clean id; a
            # non-ASCII display name falls back to a deterministic hash id.
            raw_id = project_id_from_display_name(name)
        if not raw_id:
            return JSONResponse({"error": "id or name is required"}, status_code=400)
        attach_path = str(body.get("path") or "").strip()
        git_url = str(body.get("git_url") or "").strip()
        with_workspace = bool(body.get("with_workspace"))
        if sum(1 for flag in (bool(attach_path), bool(git_url), with_workspace) if flag) > 1:
            return JSONResponse(
                {"error": "choose ONE source: path= (attach) | git_url= (clone) | with_workspace (genesis)"},
                status_code=400,
            )
        drive_root = request_drive_root(request)
        repo_dir = request_repo_dir(request)

        # An EXISTING id + a requested source is a conflict, checked BEFORE any
        # clone/validation side effect (adversarial r1): silently re-sourcing an
        # existing row would leave the registry lying (clone_url=B, working_dir=A)
        # and the fresh clone dangling. Source-less create stays idempotent.
        from ouroboros.projects_registry import get_project

        _existing = get_project(drive_root, sanitize_project_id(raw_id))
        if _existing and (attach_path or git_url or with_workspace):
            return JSONResponse(
                {
                    "error": (
                        f"project {sanitize_project_id(raw_id)!r} already exists"
                        " — pick another name/id (re-sourcing an existing project is not supported)"
                    ),
                    "error_code": "project_exists",
                },
                status_code=409,
            )

        working_dir, provenance, clone_url = "", "none", ""
        init_git_skipped: list = []
        if attach_path:
            from ouroboros.project_sources import attach_snapshot_init, validate_attach_path

            # Off the event loop: the guard forks `git` with a 5 s timeout, and a
            # repository that stalls it would otherwise freeze every other request
            # for those 5 seconds. The init-git route already runs it this way.
            resolved, error = await asyncio.to_thread(
                validate_attach_path, attach_path, system_repo_dir=repo_dir, drive_root=drive_root
            )
            if error:
                return JSONResponse({"error": error}, status_code=400)
            if bool(body.get("init_git")):
                init_error, init_git_skipped = await asyncio.to_thread(attach_snapshot_init, resolved)
                if init_error:
                    return JSONResponse({"error": f"init_git failed: {init_error}"}, status_code=400)
            # A11/A12: an UNTRACKED folder is admitted and stays untracked. Attach used
            # to refuse it outright because task admission demanded a git worktree root,
            # which made "designate a place" and "put that place under git" one
            # inseparable decision taken on the owner's behalf. They are now two: the
            # project keeps the folder, and admission raises the typed
            # `git_init_required` offer before the FIRST file task instead. The
            # remaining validate_attach_path guards are untouched.
            working_dir, provenance = str(resolved), "attached"
        elif git_url:
            from ouroboros.project_sources import clone_project_repo

            cloned, code, detail = await asyncio.to_thread(clone_project_repo, git_url, raw_id)
            if code:
                status = 401 if code == "auth_required" else 400
                return JSONResponse({"error": detail, "error_code": code}, status_code=status)
            working_dir, provenance, clone_url = cloned, "cloned", git_url

        entry = create_project(
            drive_root,
            sanitize_project_id(raw_id),
            name=name,
            working_dir=working_dir,
            origin="owner_ui",
        )
        if with_workspace:
            workspace = ensure_project_workspace(drive_root, entry["id"], repo_dir)
            if workspace:
                working_dir, provenance = workspace, "genesis"
        if working_dir and not str(entry.get("working_dir") or "").strip():
            # create_project was idempotent for an existing row — bind the folder now.
            # ATOMICALLY: the read above and this write are two separately-locked
            # operations, so testing `entry` and then overwriting is precisely the
            # read-then-write race DEVELOPMENT.md forbids. `set_working_dir_if_absent`
            # re-tests under the same lock that writes, and a concurrent binder keeps
            # its folder instead of losing it here.
            set_working_dir_if_absent(drive_root, entry["id"], working_dir)
        if _existing and provenance == "none":
            # Source-less repeat create of an EXISTING project is a pure idempotent
            # lookup: provenance/clone_url/trusted_at are ADDITIVE HISTORICAL FACTS
            # (registry docstring + ARCHITECTURE) and must not be clobbered to
            # "none" (triad r1 scope critical: a folder-bearing attached project
            # would be relabeled provenance=none).
            return JSONResponse({"project": entry})
        stamped = update_project(
            drive_root, entry["id"],
            provenance=provenance,
            clone_url=clone_url,
            trusted_at=utc_now_iso() if provenance in ("attached", "cloned") else str(entry.get("trusted_at") or ""),
        )
        payload: dict = {"project": stamped or entry}
        if init_git_skipped:
            # Disclosed omission (P1): credential-shaped files excluded from the
            # attach snapshot; they stay untracked via .git/info/exclude.
            payload["init_git_skipped"] = init_git_skipped[:50]
        # Other open tabs learn of the new project immediately, matching the
        # update/delete/promote siblings (scope r6: creation relied on the 20s poll).
        _broadcast_projects_changed(str(entry.get("id") or ""), entry.get("chat_id"))
        return JSONResponse(payload)
    except Exception as exc:
        return json_exception(exc)


async def api_project_update(request: Request) -> JSONResponse:
    """POST /api/projects/{project_id}/update — rename (the only mutable UI field)."""
    try:
        from ouroboros.projects_registry import PROJECT_NAME_MAX, get_project, update_project

        project_id = str(request.path_params.get("project_id") or "").strip()
        body = await request.json()
        if not isinstance(body, dict):
            return JSONResponse({"error": "body must be a JSON object"}, status_code=400)
        drive_root = request_drive_root(request)
        if get_project(drive_root, project_id) is None:
            return JSONResponse({"error": f"unknown project: {project_id}"}, status_code=404)
        name = str(body.get("name") or "").strip()
        if not name:
            return JSONResponse({"error": "name is required"}, status_code=400)
        if len(name) > PROJECT_NAME_MAX:
            return JSONResponse(
                {"error": f"name must be <= {PROJECT_NAME_MAX} characters"},
                status_code=400,
            )
        entry = update_project(drive_root, project_id, name=name)
        _broadcast_projects_changed(str((entry or {}).get("id") or project_id), (entry or {}).get("chat_id"))
        return JSONResponse({"project": entry})
    except Exception as exc:
        return json_exception(exc)


async def api_project_init_git(request: Request) -> JSONResponse:
    """POST /api/projects/{project_id}/init-git — the owner's YES to the typed
    ``git_init_required`` offer (A12).

    This is the ONLY thing that answer calls, and it runs the SAME
    ``attach_snapshot_init`` the create dialog's ``init_git`` runs — one snapshot
    commit of what is already in the folder, credential-shaped files deliberately
    left untracked and disclosed. Admission never reaches this route by itself:
    it raises the offer and stops, and the owner decides.

    The attach guards are re-run against the CURRENT working_dir rather than
    trusted from registration time — the registry is a file on disk, and the one
    thing this route does is write into a folder, so it re-establishes that the
    folder is still a real directory outside the Ouroboros repo/data roots first.
    """
    try:
        import asyncio

        from ouroboros.project_sources import attach_snapshot_init, validate_attach_path
        from ouroboros.projects_registry import get_project

        project_id = str(request.path_params.get("project_id") or "").strip()
        drive_root = request_drive_root(request)
        repo_dir = request_repo_dir(request)
        project = get_project(drive_root, project_id)
        if project is None:
            return JSONResponse({"error": f"unknown project: {project_id}"}, status_code=404)
        working_dir = str(project.get("working_dir") or "").strip()
        if not working_dir:
            return JSONResponse(
                {
                    "error": (
                        f"project {project['id']!r} has no working folder to initialise — "
                        "attach or create one first"
                    ),
                    "error_code": "no_working_dir",
                },
                status_code=400,
            )
        resolved, error = await asyncio.to_thread(
            validate_attach_path, working_dir, system_repo_dir=repo_dir, drive_root=drive_root
        )
        if error:
            return JSONResponse({"error": f"working folder is unusable: {error}"}, status_code=400)
        init_error, skipped = await asyncio.to_thread(attach_snapshot_init, resolved)
        if init_error:
            return JSONResponse({"error": f"init_git failed: {init_error}"}, status_code=400)
        payload: dict = {"project": project, "working_dir": str(resolved)}
        if skipped:
            # Disclosed omission (P1): credential-shaped files stayed out of the
            # snapshot and remain untracked via .git/info/exclude.
            payload["init_git_skipped"] = skipped[:50]
        _broadcast_projects_changed(str(project["id"]), project.get("chat_id"))
        return JSONResponse(payload)
    except Exception as exc:
        return json_exception(exc)


async def api_project_delete(request: Request) -> JSONResponse:
    """Fence admission, cancel the live tree, then preserve a tombstone.

    The response acknowledges that deletion has STARTED; cancellation runs off
    the event loop because cancelling a running task may join/respawn a worker.
    Chat, folder, history, memory, id, and immutable bindings are never removed.

    Its threads' CHECKOUTS are a different matter, and this route used to walk
    past them entirely (I1). Every clause ``api_thread_delete`` gives for taking a
    thread's checkout with the thread is equally true of the PROJECT: a tombstoned
    project is invisible on every surface, ``list_thread_worktrees`` has no route,
    and branch/merge refuse a thread that is not live — so a checkout left behind
    is a folder and a ``thread/*`` branch that A10's explicit removal can no
    longer reach, on durable state exempt from every GC. Nothing applied it, so
    one gesture destroyed-by-orphaning a file that existed only inside a checkout,
    and D4's "a thread's worktree is NEVER removed silently" had a hole exactly
    one click wide.

    So: BEFORE the fence, refuse ``threads_hold_checkouts`` when any checkout
    holds work that cannot be rebuilt — the same ``checkout_work_at_risk`` judge
    thread deletion uses and the same sentence, naming the threads and the
    explicit removal route. Asked before anything is fenced, because a refusal
    must leave the project exactly as it was. Otherwise the checkouts go WITH the
    project and are disclosed on the answer. A checkout the removal cannot take
    yet (a task is still writing in that folder) is reported as PENDING and swept
    by the cancellation worker once the project quiesces, so no path leaves an
    orphan silently.

    "Swept" is not "force-removed": the sweep re-asks ``checkout_work_at_risk`` per
    checkout, because the inspection this route took is a fact about a moment that
    has passed by then — the task that made a removal refuse ``project_busy`` here
    can commit work and edit tracked files before it stops. A checkout that became
    at-risk in that window survives the sweep and is disclosed on the tombstoned
    row (``delete_error``) and in a chat note naming the folder and its branch (P1).
    """
    try:
        import asyncio

        from ouroboros.projects_registry import (
            PROJECT_TOMBSTONED,
            begin_project_deletion,
            get_reserved_project,
        )
        from ouroboros.thread_worktrees import (
            project_checkouts_at_risk,
            remove_project_thread_worktrees,
        )
        from supervisor.task_lifecycle import start_project_deletion

        project_id = str(request.path_params.get("project_id") or "").strip()
        drive_root = request_drive_root(request)
        entry = get_reserved_project(drive_root, project_id)
        if entry is None:
            return JSONResponse({"error": f"unknown project: {project_id}"}, status_code=404)
        # All queue/binding comparisons use the canonical registry id.  The
        # lookup accepts a case-variant for compatibility, but cancellation must
        # not compare that raw route token against canonical task.project_id.
        project_id = str(entry.get("id") or project_id)
        at_risk = await asyncio.to_thread(project_checkouts_at_risk, drive_root, project_id)
        if at_risk:
            return JSONResponse(
                {
                    "ok": False,
                    "reason": "threads_hold_checkouts",
                    "message": _project_checkouts_refusal_message(at_risk),
                    "project_id": project_id,
                    "threads": [
                        {
                            "thread_id": item["thread_id"],
                            "path": item["path"],
                            "branch": item["branch"],
                            "inspection": item["inspection"],
                        }
                        for item in at_risk
                    ],
                },
                status_code=409,
            )
        fenced = begin_project_deletion(drive_root, project_id)
        if fenced is None:
            return JSONResponse({"error": f"unknown project: {project_id}"}, status_code=404)
        chat_id = fenced.get("chat_id")
        # Inside the fence: routing into the project is already closed, so nothing
        # NEW can start writing in a checkout while it is being taken.
        swept = await asyncio.to_thread(remove_project_thread_worktrees, drive_root, project_id)
        _broadcast_projects_changed(project_id, chat_id)
        if str(fenced.get("lifecycle") or "") != PROJECT_TOMBSTONED:
            start_project_deletion(drive_root, project_id, chat_id)
        return JSONResponse({
            "ok": True,
            "project_id": project_id,
            "folder_untouched": True,
            "worktrees_removed": swept["removed"],
            "branches_removed": swept["branches"],
            # Not removable YET (a task is still in that folder). The cancellation
            # worker takes them once the project quiesces — unless the work in them
            # has become unrebuildable by then, in which case they SURVIVE and the
            # tombstone discloses them. Named here, with their folder and branch, so
            # "the checkouts went with it" is never claimed before it is true.
            "worktrees_pending": swept["kept"],
        })
    except Exception as exc:
        return json_exception(exc)


def _project_checkouts_refusal_message(at_risk: list) -> str:
    """Why a project delete stops, naming the threads whose work is at stake.

    Built from the SAME ``_delete_refusal_message`` a single thread's deletion
    uses, so the two gestures explain the identical fact identically — a second
    copy would drift the moment either was edited.
    """
    from ouroboros.gateway.project_threads import _delete_refusal_message

    count = len(at_risk)
    head = (
        f"{count} of this project's threads {'has' if count == 1 else 'have'} a checkout "
        "holding work that exists nowhere else. Deleting the project would delete those "
        "folders and their branches, and a deleted project leaves no surface that could "
        "reach them — so the delete stops here."
    )
    lines = [
        f"Thread {item['thread_id']}: {_delete_refusal_message(item['risk'])}"
        for item in at_risk
    ]
    return " ".join([head, *lines])


async def _thread_body(request: Request) -> Any:
    try:
        body = await request.json()
    except Exception:
        body = {}
    return body if isinstance(body, dict) else None


def _thread_name_error(name: str, *, required: bool) -> Any:
    from ouroboros.projects_registry import THREAD_NAME_MAX

    if required and not name:
        return JSONResponse({"error": "name is required"}, status_code=400)
    if len(name) > THREAD_NAME_MAX:
        return JSONResponse(
            {"error": f"name must be <= {THREAD_NAME_MAX} characters"}, status_code=400
        )
    return None


async def api_project_thread_create(request: Request) -> JSONResponse:
    """POST /api/projects/{project_id}/threads — a new empty thread.

    Owner surface only (gateway route, not an LLM-callable tool). The new
    thread's chat id rides the `projects_changed` broadcast so every open client
    adds it to its known-chat set BEFORE a live frame for it can arrive.
    """
    # Imported BEFORE the guard, not inside it: an `except` clause naming an
    # unbound local would turn any earlier failure into a NameError.
    from ouroboros.project_threads_registry import ThreadLifecycleError

    try:
        from ouroboros.projects_registry import create_thread, get_project, touch_project

        project_id = str(request.path_params.get("project_id") or "").strip()
        body = await _thread_body(request)
        if body is None:
            return JSONResponse({"error": "body must be a JSON object"}, status_code=400)
        drive_root = request_drive_root(request)
        project = get_project(drive_root, project_id)
        if project is None:
            return JSONResponse({"error": f"unknown project: {project_id}"}, status_code=404)
        name = str(body.get("name") or "").strip()
        invalid = _thread_name_error(name, required=False)
        if invalid is not None:
            return invalid
        thread = create_thread(drive_root, str(project["id"]), name=name)
        touch_project(drive_root, str(project["id"]))
        _broadcast_projects_changed(str(project["id"]), thread.get("chat_id"))
        return JSONResponse({"project_id": str(project["id"]), "thread": thread})
    except ThreadLifecycleError as exc:
        # A project on its way out refusing thread changes is a PRECONDITION the
        # owner can read, not a malformed request and not a crash: it answers 409
        # with the same typed reason the archive/restore/delete routes use
        # (T3R-17). Caught BEFORE ValueError because it is one.
        return JSONResponse(
            {"ok": False, "reason": exc.reason, "error": str(exc), "message": str(exc)},
            status_code=409,
        )
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    except Exception as exc:
        return json_exception(exc)


async def api_project_thread_update(request: Request) -> JSONResponse:
    """POST /api/projects/{project_id}/threads/{thread_id}/update — rename."""
    # Imported BEFORE the guard, not inside it: an `except` clause naming an
    # unbound local would turn any earlier failure into a NameError.
    from ouroboros.project_threads_registry import ThreadLifecycleError

    try:
        from ouroboros.projects_registry import get_project, get_thread, rename_thread

        project_id = str(request.path_params.get("project_id") or "").strip()
        thread_id = str(request.path_params.get("thread_id") or "").strip()
        body = await _thread_body(request)
        if body is None:
            return JSONResponse({"error": "body must be a JSON object"}, status_code=400)
        drive_root = request_drive_root(request)
        project = get_project(drive_root, project_id)
        if project is None:
            return JSONResponse({"error": f"unknown project: {project_id}"}, status_code=404)
        if get_thread(drive_root, str(project["id"]), thread_id) is None:
            return JSONResponse({"error": f"unknown thread: {thread_id}"}, status_code=404)
        name = str(body.get("name") or "").strip()
        invalid = _thread_name_error(name, required=True)
        if invalid is not None:
            return invalid
        thread = rename_thread(drive_root, str(project["id"]), thread_id, name)
        if thread is None:
            return JSONResponse({"error": f"unknown thread: {thread_id}"}, status_code=404)
        _broadcast_projects_changed(str(project["id"]), thread.get("chat_id"))
        return JSONResponse({"project_id": str(project["id"]), "thread": thread})
    except ThreadLifecycleError as exc:
        # A project on its way out refusing thread changes is a PRECONDITION the
        # owner can read, not a malformed request and not a crash: it answers 409
        # with the same typed reason the archive/restore/delete routes use
        # (T3R-17). Caught BEFORE ValueError because it is one.
        return JSONResponse(
            {"ok": False, "reason": exc.reason, "error": str(exc), "message": str(exc)},
            status_code=409,
        )
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    except Exception as exc:
        return json_exception(exc)


async def api_project_thread_fork(request: Request) -> JSONResponse:
    """POST /api/projects/{project_id}/threads/{thread_id}/fork.

    The source thread is UNTOUCHED: the new thread stores a cursor into the
    source's rows, so no history is copied and no row identity is minted twice.
    """
    # Imported BEFORE the guard, not inside it: an `except` clause naming an
    # unbound local would turn any earlier failure into a NameError.
    from ouroboros.project_threads_registry import ThreadLifecycleError

    try:
        from ouroboros.projects_registry import fork_thread, get_project, touch_project

        project_id = str(request.path_params.get("project_id") or "").strip()
        thread_id = str(request.path_params.get("thread_id") or "").strip()
        drive_root = request_drive_root(request)
        project = get_project(drive_root, project_id)
        if project is None:
            return JSONResponse({"error": f"unknown project: {project_id}"}, status_code=404)
        try:
            thread = fork_thread(drive_root, str(project["id"]), thread_id)
        except ValueError as exc:
            if "unknown thread" in str(exc):
                return JSONResponse({"error": str(exc)}, status_code=404)
            raise
        touch_project(drive_root, str(project["id"]))
        _broadcast_projects_changed(str(project["id"]), thread.get("chat_id"))
        return JSONResponse({"project_id": str(project["id"]), "thread": thread})
    except ThreadLifecycleError as exc:
        # A project on its way out refusing thread changes is a PRECONDITION the
        # owner can read, not a malformed request and not a crash: it answers 409
        # with the same typed reason the archive/restore/delete routes use
        # (T3R-17). Caught BEFORE ValueError because it is one.
        return JSONResponse(
            {"ok": False, "reason": exc.reason, "error": str(exc), "message": str(exc)},
            status_code=409,
        )
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    except Exception as exc:
        return json_exception(exc)


def _broadcast_projects_changed(project_id: str, chat_id: Any) -> None:
    try:
        from supervisor.message_bus import get_bridge

        get_bridge().broadcast({"type": "projects_changed", "project_id": project_id, "chat_id": chat_id})
    except Exception:
        log.debug("projects_changed broadcast failed for %s", project_id, exc_info=True)


async def api_fs_dirs(request: Request) -> JSONResponse:
    """GET /api/fs/dirs?path= — owner-facing SERVER-SIDE directory browser for the
    New Project attach picker (works in web/Docker where no native dialog exists).
    Lists DIRECTORIES only, confined to the owner's home tree (the same boundary the
    agent's user_files root uses), never file contents. Defaults to home."""
    try:
        import pathlib as _pathlib

        from ouroboros.tool_access import path_is_relative_to

        home = _pathlib.Path.home().resolve(strict=False)
        raw = str(request.query_params.get("path") or "").strip() or str(home)
        # Confinement is checked BEFORE any existence-dependent response (triad r4:
        # a strict resolve + 404 first made this an existence oracle for arbitrary
        # host paths — outside-home must always get the same confined error).
        base = _pathlib.Path(raw).expanduser().resolve(strict=False)
        if base != home and not path_is_relative_to(base, home):
            return JSONResponse({"error": "directory browsing is confined to the home tree"}, status_code=400)
        if not base.exists():
            return JSONResponse({"error": f"path does not exist: {raw}"}, status_code=404)
        if not base.is_dir():
            return JSONResponse({"error": f"not a directory: {raw}"}, status_code=400)
        entries = []
        try:
            children = sorted(base.iterdir(), key=lambda p: p.name.casefold())
        except PermissionError:
            return JSONResponse({"error": f"permission denied: {base}"}, status_code=403)
        for child in children:
            try:
                if not child.is_dir() or child.name.startswith("."):
                    continue
            except OSError:
                continue
            entries.append({
                "name": child.name,
                "path": str(child),
                "is_git": (child / ".git").exists(),
            })
        # base is confined to the home tree, so its parent is home or inside home.
        parent = str(base.parent) if base != home else ""
        return JSONResponse({
            "path": str(base),
            "parent": parent,
            "home": str(home),
            "dirs": entries[:500],
            # No-silent-truncation honesty: a >500-child dir tells the UI more exist.
            "truncated": len(entries) > 500,
        })
    except Exception as exc:
        return json_exception(exc)


async def api_project_from_task(request: Request) -> JSONResponse:
    """Create/get a project from an existing task and bind the task to it."""
    try:
        from ouroboros.project_facts import explicit_project_id_ok, sanitize_project_id
        from ouroboros.projects_registry import (
            PROJECT_NAME_MAX,
            adopt_task_workspace,
            bind_task_to_project,
            create_project,
            touch_project,
        )

        body = await request.json()
        if not isinstance(body, dict):
            return JSONResponse({"error": "body must be a JSON object"}, status_code=400)
        task_id = str(body.get("task_id") or "").strip()
        if not task_id:
            return JSONResponse({"error": "task_id is required"}, status_code=400)
        raw_id = str(body.get("id") or body.get("project_id") or f"task-{task_id}").strip()
        if not explicit_project_id_ok(raw_id):
            return JSONResponse(
                {"error": f"id {raw_id!r} is not filesystem-clean (lowercase alphanumeric/_/-/., <=64 chars)"},
                status_code=400,
            )
        drive_root = request_drive_root(request)
        # Auto-name from the task's own title/objective when the caller sends none
        # (the one-click convert path), so no human input and no extra LLM call
        # are needed (owner P1). An explicit name still wins. Order: explicit name ->
        # server-derived (title/objective/queue) -> the frontend's objective_hint
        # (the owner's original request, for a still in-progress DIRECT chat task
        # with no server-side source yet) -> a neutral "New project". Never the bare
        # task id — the owner explicitly does not want names surfacing as "task-…".
        supplied_name = str(body.get("name") or "").strip()
        if len(supplied_name) > PROJECT_NAME_MAX:
            return JSONResponse(
                {"error": f"name must be <= {PROJECT_NAME_MAX} characters"},
                status_code=400,
            )
        # Keep separate name and canonical-dialogue channels: the short candidate is
        # capped, while source-ref lookup receives the full owner request. This avoids
        # silently identifying only a truncated fragment of the canonical message.
        full_hint = " ".join(str(body.get("objective_hint") or "").split())
        hint = full_hint
        if len(hint) > _MAX_DERIVED_NAME:
            hint = hint[: _MAX_DERIVED_NAME - 1].rstrip() + "…"
        owner_text = _owner_request_text(drive_root, task_id, full_hint)
        # LLM-first project name (Cluster B): the owner wants a name the model coined,
        # not the heuristic "task-…". Order: explicit caller name -> a title the proactive
        # card namer already coined (reused with ZERO extra call) -> an inline bounded
        # light-model call -> the heuristic (title/objective/queue) -> the frontend hint
        # -> a neutral "New project". The async namer folds the heuristic/hint candidates
        # into its own fail-soft fallback, so a missing key / timeout never blocks convert.
        if supplied_name:
            project_name = supplied_name
            _emit_naming_reason(drive_root, task_id, project_name, "supplied")
        else:
            from ouroboros.project_naming import llm_project_name_async

            preset = _preset_suggested_name(drive_root, task_id)
            if preset:
                project_name, reason = preset, "proactive_namer"
            else:
                # A skill/system task carries no owner request text; give the namer an
                # explicit skill-derived candidate so the conversion never dead-ends at
                # the neutral "New project" (the async namer folds it into its fail-soft
                # heuristic, so a missing key / timeout still lands a real name).
                derived = _derive_project_name(drive_root, task_id)
                sys_name = _system_task_display_name(drive_root, task_id)
                llm_name = await llm_project_name_async(
                    owner_text,
                    fallback_candidates=[derived, sys_name, hint],
                    drive_root=drive_root,
                    task_id=task_id,
                )
                project_name = llm_name or sys_name or "New project"
                if not project_name or project_name == "New project":
                    reason = "anonymous_fallback"
                elif owner_text:
                    reason = "llm_or_owner_text"
                elif sys_name and project_name == sys_name:
                    reason = "system_task"
                elif derived and project_name == derived:
                    reason = "derived"
                else:
                    reason = "hint_or_fallback"
            project_name = _cap_name(project_name)
            _emit_naming_reason(drive_root, task_id, project_name, reason)
        project = create_project(
            drive_root,
            sanitize_project_id(raw_id),
            name=project_name,
            origin="task_card",
        )
        # A11: the new project inherits the folder the converted task was already
        # working in. Without this the project came out folder-less and its NEXT
        # task auto-provisioned a different empty tree, silently moving the work.
        adopted, adopt_error = adopt_task_workspace(
            drive_root,
            str(project["id"]),
            _task_workspace_root(drive_root, task_id),
            system_repo_dir=request_repo_dir(request),
        )
        if adopted:
            project = dict(project, working_dir=adopted)
        # Scope the live task to its new project's one-writer lane BEFORE the durable
        # bind. The lease + assignment read task["project_id"] from the supervisor's
        # in-memory RUNNING map and PENDING list, NOT the durable bindings — so this
        # in-memory mark, not bind_task_to_project, is the conversion's effective commit
        # point for one-writer serialization. Without it a UI conversion could let a
        # concurrent same-project task be assigned (two writers), AND a still-PENDING
        # converted task would start unscoped and miss its lane. Marking BEFORE the
        # durable bind closes the interleaving where assign_tasks runs AFTER the bind but
        # BEFORE the mark (an assign pass and mark are mutually exclusive on the same
        # queue RLock, so once the mark lands the next pass already sees the lane): the
        # bind's relative timing is irrelevant since assignment never reads it. The
        # supervisor runs in-process (a thread), so we take its queue lock and use the
        # SSOT helper shared with the in-task ensure_project_scope path. No-op if the task
        # is neither running nor pending (the durable bind alone is then correct — there
        # is no live lane to occupy).
        try:
            from ouroboros.project_lease import mark_task_project
            from ouroboros.projects_registry import project_working_dirs
            from supervisor.queue import _queue_lock, persist_queue_snapshot
            from supervisor.workers import PENDING, RUNNING

            # The project->folder map goes in because marking PINS the lane of a
            # RUNNING task: pinned without it, a task that named no folder freezes
            # (pid, "") while every later candidate for the same project resolves to
            # ("", folder) and is admitted into the same folder alongside it.
            with _queue_lock:
                marked = mark_task_project(
                    RUNNING, PENDING, task_id, str(project["id"]),
                    project_working_dirs(drive_root),
                )
            # Persist the snapshot so a still-PENDING converted task survives a restart
            # STILL scoped: restore_pending_from_snapshot rebuilds PENDING from
            # state/queue_snapshot.json (assignment reads task['project_id'] from there,
            # NOT the durable bindings), and that snapshot is otherwise only rewritten on
            # the next queue event — so without this a restart in the window would restore
            # the task unscoped. Mirrors api_task_create persisting after enqueue.
            if marked:
                persist_queue_snapshot(reason="project_from_task")
        except Exception:
            log.debug("api_project_from_task: in-memory project_id update failed for %s", task_id, exc_info=True)
        binding = bind_task_to_project(
            drive_root,
            task_id,
            str(project["id"]),
            project.get("chat_id"),
            origin=_owner_task_origin(drive_root, task_id),
        )
        touch_project(drive_root, str(project["id"]))
        # Broadcast so every open tab + the live WS fan-out learns the new project
        # immediately, instead of waiting for the periodic /api/state poll (mirrors
        # the promote path in supervisor/workers.py).
        try:
            from supervisor.message_bus import get_bridge

            get_bridge().broadcast({
                "type": "projects_changed",
                "project_id": str(project["id"]),
                "chat_id": project.get("chat_id"),
            })
        except Exception:
            log.debug("api_project_from_task: projects_changed broadcast failed", exc_info=True)
        # ProjectFromTaskResponse. Both folder facts are TYPED (contract + api_types.js
        # mirror) because the conversion succeeds either way: an untyped free-text
        # disclosure no contract described and no client read meant a conversion that
        # quietly produced a PLACELESS project looked exactly like one that worked.
        payload: dict = {"project": project, "binding": binding}
        if adopted:
            payload["working_dir"] = adopted
        if adopt_error:
            # Disclosed, non-fatal (P1): the conversion succeeded, but the folder the
            # task named is no longer adoptable and the owner should hear it rather
            # than discover a folder-less project later. Logged as well as returned —
            # a browser that drops the field must not be the only witness.
            log.warning("api_project_from_task: %s", adopt_error)
            payload["working_dir_error"] = adopt_error
        return JSONResponse(payload)
    except Exception as exc:
        return json_exception(exc)


__all__ = [
    "api_fs_dirs",
    "api_project_delete",
    "api_project_from_task",
    "api_project_init_git",
    "api_project_thread_create",
    "api_project_thread_fork",
    "api_project_thread_update",
    "api_project_update",
    "api_projects_create",
    "api_projects_list",
]

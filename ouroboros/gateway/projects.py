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


def _preset_suggested_name(drive_root: object, task_id: str) -> str:
    """The name already sitting in this task's ``suggested_name`` slot, read from
    the persisted result then the live queue. Reused by turn-into-project so the
    conversion needs no extra LLM call.

    Two producers fill that slot and this function cannot tell them apart, which
    is why its naming reason says only where the name was READ: the proactive
    card namer coins one with a model for a Main-chat turn, and headless
    admission derives one lexically from the request's first line. Empty when
    neither has run (a convert click within the first ~second). Never raises."""
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


def _explicit_task_title(drive_root: object, task_id: str) -> str:
    """Existing model-authored task title, before any naming fallback/call."""
    try:
        from ouroboros.task_results import load_task_result

        result = load_task_result(drive_root, task_id) or {}
    except Exception:
        log.debug("_explicit_task_title: load_task_result failed", exc_info=True)
        result = {}
    for source in (result, _task_from_live_queue(drive_root, task_id)):
        value = str((source or {}).get("title") or "").strip()
        if value:
            return _cap_name(value)
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
    """Durable structured telemetry for HOW a project was named (which fallback path
    fired) so a future "New project" regression is visible in events.jsonl instead of
    silent (north star: transparency). Written only where a project is really CREATED.
    Best-effort; never raises."""
    try:
        import pathlib

        from ouroboros.utils import append_jsonl, utc_now_iso

        append_jsonl(pathlib.Path(str(drive_root)) / "logs" / "events.jsonl", {
            "ts": utc_now_iso(), "type": "project_named",
            "task_id": str(task_id), "name": str(name), "reason": str(reason),
        })
    except Exception:
        log.debug("_emit_naming_reason failed", exc_info=True)


def _mark_task_lane(task_id: str, pid: str) -> str:
    """Point the live queue/lease copy of ``task_id`` at ``pid`` under the queue lock
    and persist the snapshot; returns the value the lane held BEFORE the call, so a
    refused durable bind can put it back. SSOT for every post-hoc convert path here
    (fresh bind, origin adopt, sibling claim). It shares the lane MECHANICS — queue
    lock plus snapshot — with the in-task ``ensure_project_scope`` mark so those
    cannot drift apart, but NOT the authority policy: that call passes
    ``authority="binding"`` only when the origin adopt named the project.

    The lease + assignment read ``task["project_id"]`` from the supervisor's in-memory
    RUNNING map and PENDING list, NOT the durable bindings — so this mark, not
    ``bind_task_to_project``, is a conversion's effective commit point for one-writer
    serialization, and the caller makes it BEFORE the durable bind (an assign pass and
    the mark are mutually exclusive on the queue RLock, so once the mark lands the next
    pass already sees the lane; the bind's relative timing is irrelevant because
    assignment never reads it). ``authority="binding"`` because the caller OWNS the
    binding it is about to write, so the in-memory copy follows the truth even when the
    row carries a derived id from a bare-workspace promote. The snapshot keeps a
    still-PENDING converted task scoped across a restart, and is BEST-EFFORT: raising
    after the mark landed cost the caller its ``previous`` value, so a later bind
    refusal "restored" an empty lane it never set, clearing a project the task really
    had. The main loop persists every tick anyway. No-op when the task is neither
    running nor pending — the durable bind alone is then correct."""
    from ouroboros.project_lease import mark_task_project, task_lane_project_id
    from supervisor.queue import _queue_lock, persist_queue_snapshot
    from supervisor.workers import PENDING, RUNNING

    with _queue_lock:
        previous = task_lane_project_id(RUNNING, PENDING, task_id)
        marked = mark_task_project(RUNNING, PENDING, task_id, pid, authority="binding")
    if marked:
        try:
            persist_queue_snapshot(reason="project_from_task")
        except Exception:
            log.debug("_mark_task_lane: snapshot persist failed for %s", task_id, exc_info=True)
    return previous


def _live_origin_siblings(origin_ref: Any, clicked_task_id: str) -> list:
    """Live task ids OTHER than ``clicked_task_id`` carrying the same owner-message
    origin. The live scan itself is ``projects_registry.live_origin_lanes`` — the
    /api/state binding projection reads the SAME lanes, so a card this claim does
    not reach cannot keep offering a second conversion. Never raises."""
    from ouroboros.projects_registry import live_origin_lanes, origin_key

    key = origin_key(origin_ref)
    clicked = str(clicked_task_id)
    if key is None:
        return []
    return list(dict.fromkeys(
        tid for tid, ref in live_origin_lanes() if origin_key(ref) == key and tid != clicked
    ))


def _claim_origin_siblings(
    drive_root: object, project: dict, origin_ref: Any, clicked_task_id: str,
) -> dict:
    """Bind the FRESHLY created project's live origin siblings to it, so one owner
    message cannot keep a second convertible unit. A sibling already bound elsewhere
    (an explicit ``route_to_project``/``project_name`` root) is SKIPPED — its binding
    is immutable and the model's explicit choice stays the ceiling — with its lane
    restored and a DEBUG line; the owner never sees a 4xx for it. Returns
    ``{"bound": [...], "skipped": [...]}`` for the disclosure row. Never raises."""
    from ouroboros.projects_registry import bind_task_to_project, project_id_for_task
    outcome: dict = {"bound": [], "skipped": []}
    pid = str(project.get("id") or "")
    for tid in _live_origin_siblings(origin_ref, clicked_task_id):
        try:
            existing = str(project_id_for_task(drive_root, tid) or "")
        except Exception:
            existing = ""
            log.debug("_claim_origin_siblings: binding read failed for %s", tid, exc_info=True)
        if existing:
            # Bound already: an explicit choice, or an earlier claim. Immutable either
            # way - read it BEFORE the lane mark so nothing has to be rolled back.
            outcome["skipped"].append({"task_id": tid, "reason": f"already_bound:{existing}"})
            continue
        try:
            previous = _mark_task_lane(tid, pid)
        except Exception:
            previous = ""
            log.debug("_claim_origin_siblings: lane mark failed for %s", tid, exc_info=True)
        try:
            bind_task_to_project(
                drive_root, tid, pid, project.get("chat_id"),
                origin=_owner_task_origin(drive_root, tid),
            )
            outcome["bound"].append(tid)
        except Exception as exc:
            try:
                _mark_task_lane(tid, previous)
            except Exception:
                log.debug("_claim_origin_siblings: lane restore failed for %s", tid, exc_info=True)
            outcome["skipped"].append({"task_id": tid, "reason": f"{type(exc).__name__}: {exc}"})
            log.debug("_claim_origin_siblings: %s keeps its own binding", tid, exc_info=True)
    if outcome["bound"] or outcome["skipped"]:
        try:  # durable disclosure (P1) of who else joined, and who could not
            import pathlib

            from ouroboros.utils import append_jsonl, utc_now_iso

            append_jsonl(
                pathlib.Path(str(drive_root)) / "logs" / "events.jsonl",
                {
                    "ts": utc_now_iso(),
                    "type": "project_origin_siblings_bound",
                    "task_id": str(clicked_task_id),
                    "project_id": pid,
                    **outcome,
                },
            )
        except Exception:
            log.debug("project_origin_siblings_bound row failed", exc_info=True)
    return outcome


async def api_projects_list(request: Request) -> JSONResponse:
    try:
        from ouroboros.projects_registry import (
            projects_summary,
        )

        drive_root = request_drive_root(request)
        return JSONResponse({"projects": projects_summary(drive_root, limit=200)})
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
            update_project,
        )
        from ouroboros.utils import utc_now_iso

        body = await request.json()
        if not isinstance(body, dict):
            return JSONResponse({"error": "body must be a JSON object"}, status_code=400)
        # Executable gateway ABI (ABI-3, Q7=A): type-check the declared
        # ProjectCreateRequest fields (all optional) before bespoke parsing.
        from ouroboros.gateway.contracts import ProjectCreateRequest
        from ouroboros.gateway.schema import validate_ingress

        schema_errors = validate_ingress(body, ProjectCreateRequest)
        if schema_errors:
            return JSONResponse(
                {"error": f"invalid request body: {schema_errors[0]}",
                 "schema_errors": schema_errors[:8]},
                status_code=400,
            )
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
        init_git_warnings: list = []
        if attach_path:
            from ouroboros.project_sources import (
                attach_snapshot_init,
                validate_attach_path,
            )
            from ouroboros.workspace_admission import WorkspaceRootError, validate_workspace_root

            resolved, error = validate_attach_path(
                attach_path, system_repo_dir=repo_dir, drive_root=drive_root
            )
            if error:
                return JSONResponse({"error": error}, status_code=400)
            if bool(body.get("init_git")):
                # The explicit choice may create a standalone repository inside
                # an existing one. Validate that resulting worktree geometry.
                init_error, init_git_skipped = await asyncio.to_thread(attach_snapshot_init, resolved, warnings=init_git_warnings)
                if init_error:
                    return JSONResponse({"error": f"init_git failed: {init_error}"}, status_code=400)
            try:
                resolved = await asyncio.to_thread(
                    validate_workspace_root, resolved, system_repo_dir=repo_dir, drive_root=drive_root
                )
            except WorkspaceRootError as exc:
                return JSONResponse({"error": str(exc)}, status_code=400)
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
            update_project(drive_root, entry["id"], working_dir=working_dir)
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
        if init_git_warnings:
            from ouroboros.utils import append_jsonl
            payload["init_git_warnings"] = init_git_warnings
            try:
                append_jsonl(drive_root / "logs" / "events.jsonl", {
                    "ts": utc_now_iso(), "type": "project_capture_advisory", "project_id": entry["id"],
                    "findings": init_git_warnings,
                })
            except Exception:
                log.warning("Project capture advisory could not be logged", exc_info=True)
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


async def api_project_delete(request: Request) -> JSONResponse:
    """Fence admission, cancel the live tree, then preserve a tombstone.

    The response acknowledges that deletion has STARTED; cancellation runs off
    the event loop because cancelling a running task may join/respawn a worker.
    Chat, folder, history, memory, id, and immutable bindings are never removed.
    """
    try:
        from ouroboros.projects_registry import (
            PROJECT_TOMBSTONED,
            begin_project_deletion,
            get_reserved_project,
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
        fenced = begin_project_deletion(drive_root, project_id)
        if fenced is None:
            return JSONResponse({"error": f"unknown project: {project_id}"}, status_code=404)
        chat_id = fenced.get("chat_id")
        _broadcast_projects_changed(project_id, chat_id)
        if str(fenced.get("lifecycle") or "") != PROJECT_TOMBSTONED:
            start_project_deletion(drive_root, project_id, chat_id)
        return JSONResponse({"ok": True, "project_id": project_id, "folder_untouched": True})
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
            bind_task_to_project,
            create_project,
            get_project,
            origin_claim_lock,
            project_id_for_origin,
            project_id_for_task,
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
        # An IMPLICIT conversion is the one-click owner act: the browser sends the
        # default id for this task (chat_activity.js::projectIdFromTask) or none at all
        # — and ``raw_id`` already defaults to that same ``task-<task_id>`` — so the
        # request names no particular room and the work's own project answers it. Both
        # halves go through the SERVER's own sanitizer (case and single stray characters;
        # not the client's run-collapsing or dash trimming, which today's hex/sha task ids
        # never trigger). An EXPLICIT different id is a caller naming a room.
        implicit = sanitize_project_id(raw_id) == sanitize_project_id(f"task-{task_id}")
        disclosed: list = []

        def _conflicting_binding(*, disclose: bool) -> Any:
            """The 409 answer when the task is READABLY bound to another project, None
            otherwise. Read at EVERY side-effect boundary (owner decision B4=A): naming
            can call a model, create_project mints a registry row and the lease mark
            hands it a lane - all of which used to happen before the immutable bind
            refused a task that already belonged somewhere else, and a single read taken
            before the naming await cannot see a task that bound itself during it. The
            refusal NAMES that project, so no surface has to invent an explanation (there
            is no reload affordance in the desktop shell, the Telegram mini app or the
            mobile layout). An UNREADABLE store is disclosed once and read as "no
            binding", exactly as the hot path reads it, so the conversion proceeds as it
            would for an unbound task."""
            try:
                bound = str(project_id_for_task(drive_root, task_id, strict=True) or "")
            except Exception:
                if disclose:
                    log.warning(
                        "project_binding_unreadable: convert of task %s continues as unbound",
                        task_id, exc_info=True,
                    )
                return None
            if not bound or bound == sanitize_project_id(raw_id):
                return None
            bound_name = str((get_project(drive_root, bound) or {}).get("name") or "").strip() or bound
            return JSONResponse(
                {"error": f"This task already belongs to {bound_name} (id={bound}); "
                          "open it there or start a new task."},
                status_code=409,
            )

        # The owner's message identity, read ONCE: the same value answers "which
        # project does this WORK already have" below and rides the durable bind at
        # the end. Never re-derived from content (DEVELOPMENT.md anti-pattern).
        origin = _owner_task_origin(drive_root, task_id)
        origin_ref = origin.get("ref") if isinstance(origin, dict) else None

        def _bind_and_answer(project: dict, *, adopted: bool) -> Any:
            """Mark the lane, write the durable bind, answer. Called with the claim
            lock held, after ``_claim`` has refreshed the origin."""
            pid = str(project["id"])
            try:
                previous_lane = _mark_task_lane(task_id, pid)
            except Exception:
                previous_lane = ""
                log.debug("api_project_from_task: in-memory project_id update failed for %s",
                          task_id, exc_info=True)
            try:
                binding = bind_task_to_project(
                    drive_root, task_id, pid, project.get("chat_id"), origin=origin,
                )
            except Exception:
                # The lane mark anticipates a bind the immutable store can still
                # refuse. Put the lane back where it was (its previous project, or
                # none) before answering, so the durable binding stays the one truth
                # and no lane points at a project that binds nothing.
                try:
                    _mark_task_lane(task_id, previous_lane)
                except Exception:
                    log.debug("api_project_from_task: lane restore failed for %s", task_id, exc_info=True)
                refusal = _conflicting_binding(disclose=False)
                if refusal is None:
                    raise
                return refusal
            touch_project(drive_root, pid)
            if not adopted:
                # This conversion minted the project for THIS owner message, so the
                # message's other live task ids join it now instead of each keeping a
                # convert button of its own (that second button is what minted the
                # duplicate Project).
                _claim_origin_siblings(drive_root, project, origin_ref, task_id)
            # Broadcast so every open tab + the live WS fan-out learns the project
            # immediately, instead of waiting for the periodic /api/state poll
            # (mirrors the promote path in supervisor/workers.py).
            _broadcast_projects_changed(pid, project.get("chat_id"))
            # ``adopted`` is always stated: a client that must tell "this work
            # already had a project" from "this click created one" cannot read
            # that from a key's absence, which is also how a transport that drops
            # unknown fields looks.
            return JSONResponse(
                {"project": project, "binding": binding, "adopted": bool(adopted)},
            )

        def _claim(project_name: str = "", naming_reason: str = "") -> Any:
            """The whole claim under the ONE process-local claim lock: re-read the
            authority, ADOPT the project this work already has, or — once a name is
            settled — CREATE the requested one and bind to it. ``None`` when there is
            nothing to adopt and no name has been coined yet.

            One message spawns several task ids (the direct turn that received it, the
            root it promoted); each was separately convertible, so converting the
            second one minted a second Project for one piece of work. Adoption creates
            nothing and names nothing: the task joins the project its own origin
            already names, and a stale button on a card that is already bound answers
            with its project instead of an error toast. The lock is what makes two
            cards of one message clicked in two tabs (or the desktop shell and the mini
            app) yield ONE project; the naming step, which may await a model for
            seconds, stays outside it and its result is simply discarded on adopt.

            An EXPLICIT different project id is an API caller naming a specific room,
            never the one-click owner act: it is not adopted away, and the task-keyed
            409 still answers it (P13)."""
            nonlocal origin, origin_ref
            with origin_claim_lock():
                if "absent" in origin:
                    # The ingress record may persist only after this click started (and
                    # again during the naming await): a stale absence answers "this work
                    # has no project" and mints the one the sibling already owns.
                    origin = _owner_task_origin(drive_root, task_id)
                    origin_ref = origin.get("ref")
                try:
                    bound = str(project_id_for_task(drive_root, task_id, strict=True) or "")
                    adopted = str(project_id_for_origin(drive_root, origin_ref, strict=True) or "")
                except Exception:
                    bound, adopted = "", ""
                    if not disclosed:
                        disclosed.append(True)
                        log.warning(
                            "project_binding_unreadable: convert of task %s continues as unbound",
                            task_id, exc_info=True,
                        )
                target = (bound or adopted) if implicit else ""
                project = get_project(drive_root, target) if target else None
                if project is not None:
                    return _bind_and_answer(project, adopted=True)
                if not project_name:
                    return None
                # Re-validated BEFORE the first side effect: the naming step can await
                # a model for seconds, and a running task that scopes itself in that
                # window binds durably to ANOTHER project.
                refusal = _conflicting_binding(disclose=False)
                if refusal is not None:
                    return refusal
                # Only a really CREATED project gets a naming row: an adopt discards the
                # coined name, and a row for a name nobody saw reads as a regression.
                _emit_naming_reason(drive_root, task_id, project_name, naming_reason)
                return _bind_and_answer(
                    create_project(
                        drive_root, sanitize_project_id(raw_id), name=project_name,
                        origin="task_card",
                    ),
                    adopted=False,
                )

        claimed = _claim()
        if claimed is not None:
            return claimed
        if not implicit:
            refusal = _conflicting_binding(disclose=True)
            if refusal is not None:
                return refusal
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
        # LLM-first project name (Cluster B), with no human input and no extra LLM call
        # on the one-click path (owner P1): explicit caller name -> explicit task title
        # -> a title the proactive card namer already coined (both reused with ZERO extra
        # call) -> an inline bounded light-model call -> the heuristic (title/objective/
        # queue) -> the frontend's objective_hint (the owner's original request, for a
        # still in-progress DIRECT chat task with no server-side source yet) -> a neutral
        # "New project". Never the bare task id — the owner does not want names surfacing
        # as "task-…". The async namer folds the heuristic/hint candidates into its own
        # fail-soft fallback, so a missing key / timeout never blocks convert.
        if supplied_name:
            project_name, reason = supplied_name, "supplied"
        else:
            from ouroboros.project_naming import llm_project_name_async

            explicit_title = _explicit_task_title(drive_root, task_id)
            preset = _preset_suggested_name(drive_root, task_id)
            if explicit_title:
                project_name, reason = explicit_title, "explicit_task_title"
            elif preset:
                project_name, reason = preset, "preset_suggested_name"
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
        # The naming step ran OUTSIDE the claim lock because it can await a model
        # call for seconds. Claim again: if a sibling card of this same owner
        # message won the race meanwhile, the coined name is simply discarded and
        # the task joins the project that work already has.
        return _claim(project_name or "New project", reason)
    except Exception as exc:
        return json_exception(exc)


__all__ = [
    "api_fs_dirs",
    "api_project_delete",
    "api_project_from_task",
    "api_project_update",
    "api_projects_create",
    "api_projects_list",
]

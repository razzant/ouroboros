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
from starlette.concurrency import run_in_threadpool

from ouroboros.gateway._helpers import json_exception, request_drive_root, request_repo_dir, run_sync_to_completion
from ouroboros.project_admission import ProjectAdmissionError, project_admission_basis

log = logging.getLogger(__name__)


def _task_from_live_queue(drive_root: object, task_id: str) -> dict:
    """The read-back task record of a live or pending attempt."""
    try:
        from ouroboros.queue_observation import load_queue_document

        data = load_queue_document(drive_root)  # shared task values expanded, parsed once per publication
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


def _task_conversion_sources(drive_root: object, task_id: str) -> list[dict]:
    """Records for this exact episode, not Main's latest message or a client hint."""
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

        sources.append(load_effective_task_result(drive_root, task_id, materialize_artifacts=False) or {})
    except Exception:
        log.debug("Conversion result lookup failed", exc_info=True)
    sources.append(_task_from_live_queue(drive_root, task_id))
    return sources


def _owner_task_origin(drive_root: object, task_id: str, *, sources: list[dict] | None = None) -> dict:
    """Reuse the captured origin that the placement reader already projects."""
    from ouroboros.project_dialogue import _text_sha256, owner_message_ref_is_valid

    if sources is None:
        sources = _task_conversion_sources(drive_root, task_id)
    for source in sources:
        metadata = source.get("metadata") or {}
        ref = source.get("origin_message_ref") or metadata.get("origin_message_ref")
        if owner_message_ref_is_valid(ref):
            snapshot = metadata.get("origin_message_snapshot") or {}
            text = snapshot.get("placeholder_text") or source.get("origin_message_text") or metadata.get("origin_message_text")
            if isinstance(text, str) and text.strip() and _text_sha256(text) == ref.get("text_sha256"):
                return {"ref": dict(ref), "text": text, **({"snapshot": snapshot} if snapshot else {})}
    return {"absent": "post_hoc_unresolved"}


def _conversion_origin(drive_root, task_id, sources):
    """The captured opening: carried text its hash proves, else the one stored row its exact ref and hash identify."""
    from ouroboros.owner_relay import origin_snapshot, source_words
    from ouroboros.project_dialogue import _text_sha256, owner_message_ref_is_valid, resolve_owner_message_source
    from ouroboros.store import StoreError

    origin = _owner_task_origin(drive_root, task_id, sources=sources)
    if origin.get("ref"):
        return origin
    for row in sources:
        ref = row.get("origin_message_ref") or (row.get("metadata") or {}).get("origin_message_ref")
        if not owner_message_ref_is_valid(ref):
            continue
        try:
            original = resolve_owner_message_source(drive_root, ref)
        except (OSError, StoreError):
            continue
        words = source_words(original) if original else None
        if words and words.strip() and _text_sha256(words) == ref["text_sha256"]:  # a placeholder row keeps its marks
            return {"ref": ref, "text": words, "snapshot": origin_snapshot(original, drive_root)}
    return origin


def _prepare_conversion(drive_root, task_id, project_id, supplied_name, default):
    """Read the captured opening and destination off the event loop, before naming."""
    from ouroboros.project_naming import captured_project_request, captured_project_title
    from ouroboros.project_placement import placement_basis
    from supervisor.continuation_admission import verified_work_episodes

    sources = _task_conversion_sources(drive_root, task_id)
    origin = _conversion_origin(drive_root, task_id, sources)
    basis = placement_basis(drive_root, task_id, project_id)
    if supplied_name or basis.get("project"):
        return origin, basis, supplied_name, "supplied", ""
    try:
        episodes = verified_work_episodes(drive_root, task_id)
    except ValueError as exc:
        return origin, basis, "", getattr(exc, "reason", ""), ""
    if default and len(episodes) > 1:
        basis = placement_basis(drive_root, task_id, project_id, episodes=episodes)
        if basis.get("project"):
            return origin, basis, "", "", ""
    first = episodes[0]
    first_sources = sources if first == task_id else _task_conversion_sources(drive_root, first)
    first_origin = origin if first == task_id else _conversion_origin(drive_root, first, first_sources)
    name, reason = captured_project_title(first_sources)
    text = "" if name else captured_project_request(drive_root, first, first_sources, first_origin)
    return origin if origin.get("ref") else first_origin, basis, name, reason, text


def _emit_naming_reason(drive_root: object, task_id: str, name: str, reason: str) -> None:
    """Record the title's source when this request actually created a project."""
    try:
        import pathlib

        from ouroboros.utils import append_jsonl, utc_now_iso

        append_jsonl(pathlib.Path(str(drive_root)) / "logs" / "events.jsonl", {
            "ts": utc_now_iso(), "type": "project_named",
            "task_id": str(task_id), "name": str(name), "reason": str(reason),
        })
    except Exception:
        log.debug("_emit_naming_reason failed", exc_info=True)


async def api_projects_list(request: Request) -> JSONResponse:
    try:
        from ouroboros.project_facts import explicit_project_id_ok
        from ouroboros.projects_registry import list_sidebar_projects, projects_summary

        drive_root = request_drive_root(request)
        query = request.query_params
        if "chat_id" in query:
            projects = await run_in_threadpool(projects_summary, drive_root, chat_id=query["chat_id"], by_message=True)
            return JSONResponse({"projects": projects})
        if "id" in query:
            if not explicit_project_id_ok(query["id"]):
                return JSONResponse({"projects": []})
            projects = await run_in_threadpool(projects_summary, drive_root, project_id=query["id"], by_message=True)
            return JSONResponse({"projects": projects})
        # The plain list stays capped; `all=1` names the complete list (every active and deleting Project).
        projects = await run_in_threadpool(
            lambda: projects_summary(drive_root, by_message=True, limit=None if query.get("all") == "1" else 200,
                                     sidebar_rows=list_sidebar_projects(drive_root, require_readable=True)))
        return JSONResponse({"projects": projects})
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
    - ``no_folder``   record the choice to keep none (``project_sources``), 409 if the row
                      was registered or changed meanwhile (a promoted ``source=``);
    - none of these   a file-less project whose first promoted task provisions one.

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
        no_folder = body.get("no_folder") is True
        if sum(1 for flag in (bool(attach_path), bool(git_url), with_workspace, no_folder) if flag) > 1:
            return JSONResponse(
                {"error": "choose ONE source: path= (attach) | git_url= (clone) | with_workspace (genesis) | no_folder"},
                status_code=400,
            )
        drive_root = request_drive_root(request)
        repo_dir = request_repo_dir(request)

        # An EXISTING id + a requested source is a conflict, checked BEFORE any
        # clone/validation side effect (adversarial r1): silently re-sourcing an
        # existing row would leave the registry lying (clone_url=B, working_dir=A)
        # and the fresh clone dangling. Source-less create stays idempotent.
        from ouroboros.projects_registry import get_project

        def _project_exists() -> JSONResponse:
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

        _existing = get_project(drive_root, sanitize_project_id(raw_id), strict=True)
        if _existing and (attach_path or git_url or with_workspace or no_folder):
            return _project_exists()

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
        if no_folder and not entry.get("created"):
            return _project_exists()  # registered since the check above (a promoted source=): never re-chosen here
        if with_workspace:
            workspace = ensure_project_workspace(drive_root, entry["id"], repo_dir)
            if workspace:
                working_dir, provenance = workspace, "genesis"
        if working_dir and not str(entry.get("working_dir") or "").strip():
            # create_project was idempotent for an existing row — bind the folder now.
            update_project(drive_root, entry["id"], working_dir=working_dir)
        if not entry["created"] and provenance == "none":
            # Source-less repeat create of an EXISTING project is a pure idempotent
            # lookup: provenance/clone_url/trusted_at are ADDITIVE HISTORICAL FACTS
            # (registry docstring + ARCHITECTURE) and must not be clobbered to
            # "none" (triad r1 scope critical: a folder-bearing attached project
            # would be relabeled provenance=none).
            return JSONResponse({"project": entry})
        trusted_at = utc_now_iso() if provenance in ("attached", "cloned") else str(entry.get("trusted_at") or "")
        try:  # No folder binds the row this request registered: a folder attached since refuses it, never cleared
            stamped = update_project(
                drive_root, entry["id"], provenance=provenance, clone_url=clone_url, trusted_at=trusted_at,
                **({"no_folder": {"at": utc_now_iso(), "by": "owner"},
                    "admission_basis": project_admission_basis(entry["id"], entry)} if no_folder else {}),
            )
        except ProjectAdmissionError as exc:
            return JSONResponse({"error": f"project {entry['id']!r} changed while it was being created, so No folder "
                                          "was not recorded; see its Working folder…", "reason_code": exc.reason},
                                status_code=409)
        payload: dict = {"project": {**(stamped or entry), "created": entry["created"]}}
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
    """POST /api/projects/{project_id}/update — partial name, pin or closure update."""
    try:
        from ouroboros.projects_registry import PROJECT_NAME_MAX, get_project, update_project

        project_id = str(request.path_params.get("project_id") or "").strip()
        body = await request.json()
        if not isinstance(body, dict):
            return JSONResponse({"error": "body must be a JSON object"}, status_code=400)
        drive_root = request_drive_root(request)
        if get_project(drive_root, project_id, strict=True) is None:
            return JSONResponse({"error": f"unknown project: {project_id}"}, status_code=404)
        updates = {}
        if "name" in body:
            name = str(body.get("name") or "").strip()
            if not name or len(name) > PROJECT_NAME_MAX:
                return JSONResponse({"error": f"name must be 1 to {PROJECT_NAME_MAX} characters"}, status_code=400)
            updates["name"] = name
        if "pinned" in body:
            if type(body["pinned"]) is not bool:
                return JSONResponse({"error": "pinned must be a boolean"}, status_code=400)
            updates["pinned"] = body["pinned"]
        if "closed" in body:
            if type(body["closed"]) is not bool:
                return JSONResponse({"error": "closed must be a boolean"}, status_code=400)
            from ouroboros.utils import utc_now_iso
            updates["closed"] = {"at": utc_now_iso(), "by": "owner", "note": "Closed by owner."} if body["closed"] else None
        if not updates:
            return JSONResponse({"error": "name, pinned or closed is required"}, status_code=400)
        entry = update_project(drive_root, project_id, **updates)
        if entry is None:
            return JSONResponse({"error": "project is no longer active"}, status_code=409)
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
        entry = get_reserved_project(drive_root, project_id, strict=True)
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


async def api_project_folder(request: Request) -> JSONResponse:
    """GET/POST /api/projects/{project_id}/folder — the room's working folder as ``project_folder`` sees it,
    or one change of it by the owner. A refusal answers its typed ``reason_code``; nothing was written."""
    try:
        from ouroboros import project_folder
        from ouroboros.gateway.project_folder_contracts import ProjectFolderRequest
        from ouroboros.gateway.schema import validate_ingress

        project_id = str(request.path_params.get("project_id") or "").strip()
        drive_root, repo_dir = request_drive_root(request), request_repo_dir(request)
        if request.method == "GET":
            view = await run_in_threadpool(project_folder.folder_view, drive_root, project_id, repo_dir=repo_dir)
            if view is None:
                return JSONResponse({"error": f"unknown project: {project_id}"}, status_code=404)
            return JSONResponse({"folder": view})
        body = await request.json()
        errors = validate_ingress(body, ProjectFolderRequest)
        if errors:
            return JSONResponse({"error": f"invalid request body: {errors[0]}"}, status_code=400)
        outcome = await run_in_threadpool(
            lambda: project_folder.change_folder(
                drive_root, project_id, action=body["action"], expected=body["expected"], actor="owner",
                repo_dir=repo_dir, path=str(body.get("path") or "")))
        if outcome["status"] != "changed":
            status = {"unknown_project": 404, "invalid_folder": 400}.get(outcome["reason"], 409)
            return JSONResponse({"error": outcome["detail"], "reason_code": outcome["reason"]}, status_code=status)
        view = outcome["view"]  # None: changed, but the look after it failed (``folder_error``); the change stands
        _broadcast_projects_changed(outcome["receipt"]["project_id"], view["chat_id"] if view else None)
        return JSONResponse({"folder": view, "receipt": outcome["receipt"],
                             **({"folder_error": outcome["view_error"]} if view is None else {})})
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
        from ouroboros.projects_registry import PROJECT_NAME_MAX
        from ouroboros.project_placement import conversion_destination, move_task_to_project
        from supervisor import workers

        body = await request.json()
        if not isinstance(body, dict):
            return JSONResponse({"error": "body must be a JSON object"}, status_code=400)
        task_id = str(body.get("task_id") or "").strip()
        if not task_id:
            return JSONResponse({"error": "task_id is required"}, status_code=400)
        raw_id = str(body.get("id") or body.get("project_id") or "").strip()
        if raw_id and not explicit_project_id_ok(raw_id):
            return JSONResponse(
                {"error": f"id {raw_id!r} is not filesystem-clean (lowercase alphanumeric/_/-/., <=64 chars)"},
                status_code=400,
            )
        drive_root = request_drive_root(request)
        supplied_name = str(body.get("name") or "").strip()
        if len(supplied_name) > PROJECT_NAME_MAX:
            return JSONResponse({"error": f"name must be <= {PROJECT_NAME_MAX} characters"}, status_code=400)
        project_id, default = conversion_destination(task_id, sanitize_project_id(raw_id), named=bool(supplied_name))
        origin, basis, project_name, reason, owner_text = await run_in_threadpool(
            _prepare_conversion, drive_root, task_id, project_id, supplied_name, default)
        refusal = "" if project_name else (reason or "project_name_source_unavailable")
        if owner_text:
            from ouroboros.project_naming import llm_project_name_async

            project_name = await llm_project_name_async(owner_text, drive_root=drive_root, task_id=task_id)
            refusal, reason = ("" if project_name else "project_name_unavailable"), "model_name"
        outcome = await run_sync_to_completion(
            move_task_to_project, drive_root, task_id, project_id, name=project_name, origin=origin, basis=basis,
            running=workers.RUNNING, pending=workers.PENDING, initiated_by="owner",
            automatic_name=not supplied_name, naming_refusal=refusal,
            adopt_work_room=not supplied_name and default,
        )
        if outcome["project"].get("created"):
            await run_in_threadpool(_emit_naming_reason, drive_root, task_id, project_name, reason)
        return JSONResponse(outcome)
    except ProjectAdmissionError as exc:
        return JSONResponse({"error": str(exc), "reason_code": exc.reason}, status_code=409)
    except Exception as exc:
        return json_exception(exc)


__all__ = [
    "api_fs_dirs",
    "api_project_delete",
    "api_project_folder",
    "api_project_from_task",
    "api_project_update",
    "api_projects_create",
    "api_projects_list",
]

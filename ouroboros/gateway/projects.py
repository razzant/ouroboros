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
    no server-side record yet). Used to seed the project thread with the owner's
    message on "turn into project" so the project chat reads from the request, not
    a mid-flight working bubble (C4.5). Never raises."""
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


def _owner_message_send_ts(
    drive_root: object, body: str, *, source_chat_id: int = 0, not_after: str = ""
) -> str:
    """Timestamp of the owner's inbound message THAT STARTED this task — its true send
    time, strictly earlier than any working bubble. The inbound chat row is not tagged
    with the task id (it is logged before the task exists), so it is matched by its text;
    to avoid binding to a DUPLICATE identical request from another thread/task, the match
    is confined to the task's originating ``source_chat_id`` and to rows at or before the
    task's creation (``not_after``), and the LATEST such row (closest to the task start)
    is chosen. Returns "" when none is found. Never raises."""
    want = " ".join(str(body or "").split())
    if not want:
        return ""
    try:
        import pathlib

        from ouroboros.utils import iter_jsonl_objects

        path = pathlib.Path(str(drive_root)) / "logs" / "chat.jsonl"
        if not path.exists():
            return ""
        bound = str(not_after or "").strip()
        chosen = ""
        for row in iter_jsonl_objects(path):
            if not isinstance(row, dict) or str(row.get("direction") or "") != "in":
                continue
            if source_chat_id and int(row.get("chat_id", 1) or 1) != int(source_chat_id):
                continue
            if " ".join(str(row.get("text") or "").split()) != want:
                continue
            ts = str(row.get("ts") or "").strip()
            if not ts or (bound and ts > bound):
                continue
            if ts > chosen:  # latest matching row at/before the task start
                chosen = ts
        return chosen
    except Exception:
        log.debug("_owner_message_send_ts failed", exc_info=True)
        return ""


def _mirror_owner_request_to_project_chat(
    drive_root: object, project_chat_id: int, task_id: str, text: str
) -> None:
    """Append the owner's original request to the project thread as the first
    message (C4.5). Writes a normal inbound owner row (direction="in") tagged to
    the project ``chat_id`` so history replay renders it as the owner's message at
    the top of the project chat. Best-effort: a failed mirror must never block the
    conversion. Never raises."""
    body = str(text or "").strip()
    if not body or not project_chat_id:
        return
    try:
        import pathlib

        from ouroboros.utils import append_jsonl, utc_now_iso

        # Deterministic ts so the owner's row sorts to the TOP of the project thread,
        # ahead of the working bubbles emitted while the task ran — instead of being
        # stamped 'now' at the bottom. History replay sorts purely by ts (gateway/history.py).
        # First resolve the task's creation time + originating chat (from the result, then
        # the live queue snapshot). Precedence:
        #   1) the owner's inbound message that STARTED this task — matched by body but
        #      CONFINED to the originating chat and to rows at/before task creation (so a
        #      duplicate identical request elsewhere can't bind an unrelated older ts),
        #   2) queued_at, then result ts / created_at / started_at (≈ task start),
        #   3) now (last resort).
        creation_ts = ""
        source_chat_id = 0
        try:
            from ouroboros.task_results import load_task_result

            result = load_task_result(drive_root, task_id) or {}
            live = _task_from_live_queue(drive_root, task_id) or {}
            for src in (result, live):
                for field in ("queued_at", "ts", "created_at", "started_at"):
                    val = str((src or {}).get(field) or "").strip()
                    if val and not creation_ts:
                        creation_ts = val
                if not source_chat_id:
                    try:
                        source_chat_id = int((src or {}).get("chat_id") or 0)
                    except (TypeError, ValueError):
                        source_chat_id = 0
        except Exception:
            creation_ts, source_chat_id = "", 0
        original_ts = _owner_message_send_ts(
            drive_root, body, source_chat_id=source_chat_id, not_after=creation_ts,
        ) or creation_ts
        append_jsonl(
            pathlib.Path(str(drive_root)) / "logs" / "chat.jsonl",
            {
                "ts": original_ts or utc_now_iso(),
                "direction": "in",
                "chat_id": int(project_chat_id),
                "user_id": 1,
                "text": body,
                "format": "",
                "source": "web",
                "task_id": str(task_id or ""),
            },
        )
    except Exception:
        log.debug("_mirror_owner_request_to_project_chat failed", exc_info=True)


def _mirror_final_answer_to_project_chat(
    drive_root: object, project_chat_id: int, task_id: str
) -> None:
    """v6.58.0 (§3.4a): when a task is converted to a project AFTER it already
    finished, its final answer was delivered to the ORIGINAL chat before any binding
    existed — so the new project thread would show the request with no outcome. Copy
    the terminal result text (+ artifact paths) into the project thread as an
    assistant row. Running tasks need no mirror: the send_message re-homing routes
    their final answer to the bound project chat at finalization. Never raises."""
    if not project_chat_id:
        return
    try:
        import pathlib

        from ouroboros.task_results import load_task_result
        from ouroboros.task_status import FINAL_STATUSES
        from ouroboros.utils import append_jsonl, utc_now_iso

        result = load_task_result(drive_root, task_id) or {}
        if str(result.get("status") or "").strip().lower() not in FINAL_STATUSES:
            return  # still running — the live re-homing will deliver the answer
        body = str(result.get("result") or "").strip()
        if not body:
            return
        artifact_lines: list[str] = []
        bundle = result.get("artifact_bundle") if isinstance(result.get("artifact_bundle"), dict) else {}
        for art in (bundle.get("artifacts") if isinstance(bundle.get("artifacts"), list) else [])[:10]:
            if isinstance(art, dict):
                path = str(art.get("abs_path") or art.get("path") or "").strip()
                name = str(art.get("name") or "").strip()
                if path or name:
                    artifact_lines.append(f"- {name + ': ' if name and path else ''}{path or name}")
        if artifact_lines:
            body += "\n\nArtifacts:\n" + "\n".join(artifact_lines)
        # ts: strictly AFTER the owner-request mirror (which uses the task's creation/
        # send ts) — the result's own completion ts preserves the true order.
        done_ts = str(result.get("updated_at") or result.get("ts") or "").strip() or utc_now_iso()
        append_jsonl(
            pathlib.Path(str(drive_root)) / "logs" / "chat.jsonl",
            {
                "ts": done_ts,
                "direction": "out",
                "chat_id": int(project_chat_id),
                "text": body,
                "format": "",
                "source": "project_convert_mirror",
                "task_id": str(task_id or ""),
            },
        )
    except Exception:
        log.debug("_mirror_final_answer_to_project_chat failed", exc_info=True)


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
    try:
        from ouroboros.projects_registry import projects_summary

        return JSONResponse({"projects": projects_summary(request_drive_root(request), limit=200)})
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
        from ouroboros.projects_registry import create_project, ensure_project_workspace, update_project
        from ouroboros.utils import utc_now_iso

        body = await request.json()
        if not isinstance(body, dict):
            return JSONResponse({"error": "body must be a JSON object"}, status_code=400)
        name = str(body.get("name") or "").strip()
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
            from ouroboros.project_sources import (
                attach_snapshot_init,
                is_git_worktree_root,
                validate_attach_path,
            )

            resolved, error = validate_attach_path(
                attach_path, system_repo_dir=repo_dir, drive_root=drive_root
            )
            if error:
                return JSONResponse({"error": error}, status_code=400)
            if bool(body.get("init_git")):
                init_error, init_git_skipped = await asyncio.to_thread(attach_snapshot_init, resolved)
                if init_error:
                    return JSONResponse({"error": f"init_git failed: {init_error}"}, status_code=400)
            elif not await asyncio.to_thread(is_git_worktree_root, resolved):
                # Task admission requires a git worktree root — registering a non-git
                # folder would create a project whose room tasks are born dead
                # (triad r5). Actionable refusal BEFORE any registry mutation.
                return JSONResponse(
                    {
                        "error": (
                            f"{resolved} is not a git repository — enable init_git "
                            "(makes an attach-snapshot commit) or pick a git worktree root"
                        ),
                        "error_code": "attach_requires_git",
                    },
                    status_code=400,
                )
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
        from ouroboros.projects_registry import get_project, update_project

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
        entry = update_project(drive_root, project_id, name=name)
        _broadcast_projects_changed(str((entry or {}).get("id") or project_id), (entry or {}).get("chat_id"))
        return JSONResponse({"project": entry})
    except Exception as exc:
        return json_exception(exc)


async def api_project_delete(request: Request) -> JSONResponse:
    """POST /api/projects/{project_id}/delete — unregister + unbind. The working
    folder and per-project memory store are NOT touched (delete never destroys
    owner data)."""
    try:
        from ouroboros.projects_registry import delete_project, get_project

        project_id = str(request.path_params.get("project_id") or "").strip()
        drive_root = request_drive_root(request)
        entry = get_project(drive_root, project_id)
        if entry is None:
            return JSONResponse({"error": f"unknown project: {project_id}"}, status_code=404)
        removed = delete_project(drive_root, project_id)
        _broadcast_projects_changed(project_id, entry.get("chat_id"))
        return JSONResponse({"ok": bool(removed), "project_id": project_id, "folder_untouched": True})
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
            bind_task_to_project,
            create_project,
            project_binding_for_task,
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
        # v6.58.0 (§3.4 truncation fix): TWO channels for the frontend hint. The NAME
        # candidate is capped at _MAX_DERIVED_NAME (a project name is short); the chat
        # MIRROR gets the owner's FULL request — the old single truncated channel put
        # "Сделай html сайтик … в…" (60 chars + ellipsis) into the project thread as
        # if that were the whole ask, silently losing the requirements (P1).
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
            _emit_naming_reason(drive_root, task_id, project_name, reason)
        # Was this task already converted? A repeat call (double broadcast, retry)
        # must not append the owner's request to the project thread twice (C4.5).
        first_conversion = project_binding_for_task(drive_root, task_id) is None
        project = create_project(
            drive_root,
            sanitize_project_id(raw_id),
            name=project_name,
            origin="task_card",
        )
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
            from supervisor.queue import _queue_lock, persist_queue_snapshot
            from supervisor.workers import PENDING, RUNNING

            with _queue_lock:
                marked = mark_task_project(RUNNING, PENDING, task_id, str(project["id"]))
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
        binding = bind_task_to_project(drive_root, task_id, str(project["id"]), project.get("chat_id"))
        touch_project(drive_root, str(project["id"]))
        # Seed the project thread with the owner's original request as its first
        # message, so the project chat reads from what the owner asked rather than a
        # mid-flight working bubble (C4.5). Subagent/parent progress re-homes to this
        # thread by lineage (project_chat_for_task_tree); only the owner row is copied.
        if first_conversion:
            try:
                proj_chat = int(project.get("chat_id") or 0)
            except (TypeError, ValueError):
                proj_chat = 0
            _mirror_owner_request_to_project_chat(
                drive_root, proj_chat, task_id, owner_text
            )
            # §3.4a: an ALREADY-FINISHED task's answer (+ artifact paths) is copied too,
            # so the new project thread shows request → outcome, not a dangling request.
            _mirror_final_answer_to_project_chat(drive_root, proj_chat, task_id)
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
        return JSONResponse({"project": project, "binding": binding})
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

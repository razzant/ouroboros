"""HTTP surface of skill sources, the catalog and the install lifecycle (contracts: skill_source_contracts).

Reads come from local data: the Skills page never waits for a repository, and a stale or explicitly
refreshed catalog starts a background fetch. Install and update answer promptly with the lifecycle job
that runs them (``GET /api/skills/lifecycle-queue`` follows it); a review install starts its review job
after landing. Source changes are owner writes of ``OUROBOROS_SKILL_SOURCES`` through the settings seam.
"""

from __future__ import annotations

import asyncio
import logging
import pathlib
from typing import Any, Dict, Set

from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from ouroboros.gateway._helpers import (
    json_error,
    json_exception,
    request_drive_root,
    request_json_or,
    request_repo_dir,
)
from ouroboros.skill_lifecycle_queue import (
    JobHandle,
    LifecycleJobOptions,
    run_blocking_preserving_cancellation,
    run_lifecycle_job,
)
from ouroboros.skill_source_install import SkillInstallError
from ouroboros import skill_lifecycle_queue

log = logging.getLogger(__name__)
_BACKGROUND: Set[asyncio.Task] = set()


def _refusal(exc: SkillInstallError) -> JSONResponse:
    return JSONResponse(exc.payload(), status_code=exc.status)


async def _start_job(kind: str, target: str, message: str, work: Any, after: Any = None) -> str:
    """Run ``work(progress)`` in the lifecycle lane in the background; answer its job id."""
    handle = JobHandle()

    async def _runner() -> Any:
        return await run_blocking_preserving_cancellation(work, handle.set, log_label=f"skill {kind} lifecycle operation")

    def _error(result: Any) -> str:
        return str(result.get("error") or "") if isinstance(result, dict) and result.get("ok") is False else ""

    def _message(result: Any) -> str:
        # The payload landed even when its packages did not: the job succeeds and says so in one line.
        acceptance = (result.get("acceptance") or {}) if isinstance(result, dict) else {}
        deps_error = str(acceptance.get("deps_error") or "")
        done = f"{kind} {result.get('name', target) if isinstance(result, dict) else target}: done"
        if deps_error:
            return f"{done}; its packages did not install: {deps_error.splitlines()[0][:200]}"
        return f"{done}; {acceptance['setup_error']}" if acceptance.get("setup_error") else done

    async def _job() -> None:
        try:
            result = await run_lifecycle_job(
                kind=kind, target=target, source="skills", message=message, runner=_runner,
                options=LifecycleJobOptions(progress_target=handle, result_error=_error, result_message=_message),
            )
            if after is not None:
                await asyncio.to_thread(after, result)
        except SkillInstallError as exc:
            log.info("skill %s %s refused: %s", kind, target, exc)
        except Exception:
            log.warning("skill %s %s failed", kind, target, exc_info=True)

    task = asyncio.create_task(_job())
    _BACKGROUND.add(task)
    task.add_done_callback(_BACKGROUND.discard)
    waiter = asyncio.ensure_future(handle.bound.wait())
    await asyncio.wait({waiter, task}, timeout=skill_lifecycle_queue.JOB_ANSWER_WAIT_SEC, return_when=asyncio.FIRST_COMPLETED)
    waiter.cancel()
    return handle.job_id


# --- sources -----------------------------------------------------------------------------------------

def _sources_payload(drive_root: pathlib.Path, repo_dir: pathlib.Path) -> Dict[str, Any]:
    from ouroboros.config import load_settings
    from ouroboros.skill_source_catalog import snapshot_for
    from ouroboros.skill_sources import SOURCES_SETTING, load_sources, sources_problem

    settings = load_settings()
    rows = load_sources(settings)
    out = []
    for row in rows:
        _snapshot, status = snapshot_for(drive_root, row, repo_dir=repo_dir)
        out.append({**row, **status})
    return {"sources": out, "materialized": settings.get(SOURCES_SETTING) not in (None, "", []),
            "problem": sources_problem(settings)}


async def api_skill_sources(request: Request) -> JSONResponse:
    try:
        return JSONResponse(await asyncio.to_thread(_sources_payload, request_drive_root(request),
                                                    request_repo_dir(request)))
    except Exception as exc:
        return json_exception(exc, context="skill sources read failed")


async def api_skill_source_probe(request: Request) -> JSONResponse:
    from ouroboros.skill_source_catalog import probe_source

    body = await request_json_or(request, {}, exceptions=(Exception,))
    body = body if isinstance(body, dict) else {}
    url, path = str(body.get("url") or "").strip(), str(body.get("path") or "").strip()
    if bool(url) == bool(path):
        return json_error("send either url or path", 400)
    return JSONResponse(await asyncio.to_thread(probe_source, request_drive_root(request), url=url, path=path))


def _write_sources(request: Request, change: Any, audit: Dict[str, Any]) -> JSONResponse:
    from ouroboros.gateway.owner_settings import owner_audit
    from ouroboros.skill_sources import SkillSourceError, load_sources, save_sources

    try:
        rows, touched = change(load_sources())
        saved = save_sources(rows)
    except SkillSourceError as exc:
        return JSONResponse({"ok": False, "code": "invalid_source", "error": str(exc)}, status_code=400)
    owner_audit(request, "skill_sources", audit)
    payload: Dict[str, Any] = {"ok": True, "sources": saved}
    if touched is not None:
        payload["source"] = next((row for row in saved if row["id"] == touched), None) or {}
    return JSONResponse(payload)


async def _settings_write(request: Request, fn: Any) -> JSONResponse:
    from ouroboros.gateway.settings import _run_settings_writer

    return await _run_settings_writer(lambda req, _body: fn(req), request, None)


async def api_skill_source_add(request: Request) -> JSONResponse:
    from ouroboros.skill_source_catalog import probe_source
    from ouroboros.skill_sources import with_added_source

    body = await request_json_or(request, {}, exceptions=(Exception,))
    body = body if isinstance(body, dict) else {}
    url, path = str(body.get("url") or "").strip(), str(body.get("path") or "").strip()
    name, trust = str(body.get("name") or "").strip(), str(body.get("trust") or "").strip()
    if bool(url) == bool(path) or not name or trust not in {"trusted", "review"}:
        return json_error("send url or path, a name and trust (trusted or review)", 400)
    drive_root = request_drive_root(request)
    probe = await asyncio.to_thread(probe_source, drive_root, url=url, path=path)
    if probe.get("error"):
        return JSONResponse({"ok": False, "code": "unreadable_source", "error": probe["error"]}, status_code=400)
    kind = str(probe["kind"])

    def change(rows: Any) -> Any:
        rows, row = with_added_source(rows, kind=kind, url=url, path=path, name=name, trust=trust)
        return rows, row["id"]

    response = await _settings_write(request, lambda req: _write_sources(
        req, change, {"change": "add", "kind": kind, "trust": trust}))
    _refresh_after_change(drive_root, response)
    return response


def _refresh_after_change(drive_root: pathlib.Path, response: JSONResponse) -> None:
    import json

    from ouroboros.skill_source_catalog import refresh_source
    from ouroboros.skill_source_git import refresh_in_background

    try:
        payload = json.loads(response.body)
        row = payload.get("source") or {}
        if payload.get("ok") and row.get("kind") in {"hub", "git"}:
            refresh_in_background(drive_root, row, refresh_source)
    except Exception:
        log.debug("source refresh after change failed", exc_info=True)


async def api_skill_source_change(request: Request) -> JSONResponse:
    from ouroboros.skill_sources import with_changed_source

    source_id = str(request.path_params.get("source_id") or "")
    body = await request_json_or(request, {}, exceptions=(Exception,))
    body = body if isinstance(body, dict) else {}
    trust = body.get("trust")
    if trust is not None and trust not in {"trusted", "review"}:
        return json_error("trust is trusted or review", 400)
    if "confirmed" in body and not isinstance(body.get("confirmed"), bool):
        return json_error("confirmed must be a boolean", 400)
    changes = {key: body.get(key) for key in ("name", "trust", "confirmed") if key in body}
    return await _settings_write(request, lambda req: _write_sources(
        req, lambda rows: (with_changed_source(rows, source_id, **changes), source_id),
        {"change": "edit", "source": source_id, **{k: v for k, v in changes.items() if k != "name"}}))


async def api_skill_source_remove(request: Request) -> JSONResponse:
    from ouroboros.skill_sources import without_source

    source_id = str(request.path_params.get("source_id") or "")
    return await _settings_write(request, lambda req: _write_sources(
        req, lambda rows: (without_source(rows, source_id), None), {"change": "remove", "source": source_id}))


async def api_skill_source_refresh(request: Request) -> JSONResponse:
    from ouroboros.skill_source_catalog import refresh_source, snapshot_for
    from ouroboros.skill_source_git import refresh_in_background
    from ouroboros.skill_sources import load_sources, source_by_id

    drive_root = request_drive_root(request)
    row = source_by_id(await asyncio.to_thread(load_sources), str(request.path_params.get("source_id") or ""))
    if row is None:
        return json_error("no such source", 404)
    started = row["kind"] in {"hub", "git"} and refresh_in_background(drive_root, row, refresh_source)
    _snapshot, status = await asyncio.to_thread(snapshot_for, drive_root, row, repo_dir=request_repo_dir(request))
    return JSONResponse({"ok": True, "source": {**row, **status, "refreshing": bool(started or status.get("refreshing"))}})


# --- catalog -----------------------------------------------------------------------------------------

def _catalog(drive_root: pathlib.Path, repo_dir: pathlib.Path, refresh: bool) -> Dict[str, Any]:
    from ouroboros.skill_source_catalog import catalog_view
    from ouroboros.skill_starter import read_marker

    view = catalog_view(drive_root, refresh=refresh, repo_dir=repo_dir)
    marker = read_marker(drive_root)
    view["starter"] = {"state": str(marker.get("state") or "")} if marker else {}
    return view


async def api_skill_catalog(request: Request) -> JSONResponse:
    refresh = str(request.query_params.get("refresh") or "") in {"1", "true", "yes"}
    try:
        return JSONResponse(await asyncio.to_thread(_catalog, request_drive_root(request), request_repo_dir(request),
                                                    refresh))
    except Exception as exc:
        return json_exception(exc, context="skill catalog read failed")


def _search(drive_root: pathlib.Path, repo_dir: pathlib.Path, query: str, source: str) -> Dict[str, Any]:
    from ouroboros.skill_source_catalog import catalog_view, registry_items
    from ouroboros.skill_sources import load_sources

    rows = load_sources()
    view = catalog_view(drive_root, rows=rows, refresh=True, repo_dir=repo_dir)
    words = [word for word in query.lower().split() if word]
    items = [item for item in view["items"] if (not source or item["source_id"] == source) and all(
        word in f"{item['name']} {item.get('description', '')}".lower() for word in words)]
    out: Dict[str, Any] = {"items": items}
    for row in rows:
        if row["kind"] == "registry" and query.strip() and (not source or source == row["id"]):
            try:
                items.extend(registry_items(row, query.strip()))
            except Exception as exc:
                out["registry_error"] = f"{row['name']}: {exc}"[:300]
    return out


async def api_skill_catalog_search(request: Request) -> JSONResponse:
    query = str(request.query_params.get("q") or "")
    source = str(request.query_params.get("source") or "")
    try:
        return JSONResponse(await asyncio.to_thread(_search, request_drive_root(request), request_repo_dir(request),
                                                    query, source))
    except Exception as exc:
        return json_exception(exc, context="skill catalog search failed")


async def api_skill_catalog_item(request: Request) -> JSONResponse:
    from ouroboros.skill_source_catalog import item_detail

    item_id = f"{request.path_params.get('source_id') or ''}/{request.path_params.get('name') or ''}"
    try:
        return JSONResponse(await asyncio.to_thread(item_detail, request_drive_root(request), item_id,
                                                    repo_dir=request_repo_dir(request)))
    except LookupError as exc:
        return json_error(str(exc), 404)
    except Exception as exc:
        return json_exception(exc, context="skill catalog item read failed")


def _after_install(drive_root: pathlib.Path, repo_dir: pathlib.Path) -> Any:
    from ouroboros.skill_source_install import start_review

    def after(result: Dict[str, Any]) -> None:
        if result.get("needs_review") and result.get("name"):
            start_review(drive_root, result["name"], repo_dir=repo_dir)

    return after


async def api_skill_catalog_install(request: Request) -> JSONResponse:
    from ouroboros.skill_source_install import install_item, plan_install

    body = await request_json_or(request, {}, exceptions=(Exception,))
    body = body if isinstance(body, dict) else {}
    item_id = str(body.get("id") or "").strip()
    if "/" not in item_id:
        return json_error("id must be <source_id>/<name>", 400)
    replace = body.get("replace") is True
    turn_on = body.get("turn_on_when_ready") is not False
    drive_root, repo_dir = request_drive_root(request), request_repo_dir(request)
    try:
        plan = await asyncio.to_thread(plan_install, drive_root, item_id, replace=replace, repo_dir=repo_dir)
    except SkillInstallError as exc:
        return _refusal(exc)
    name = plan["name"] or item_id.partition("/")[2]
    if plan["already_installed"]:
        return JSONResponse({"ok": True, "job_id": "", "name": name, "basis": plan["basis"], "already_installed": True})
    job_id = await _start_job(
        "install", name, f"Installing {name} from {plan['row']['name']}",
        lambda progress: install_item(drive_root, item_id, replace=replace, actor="owner",
                                      turn_on_when_ready=turn_on, repo_dir=repo_dir, progress=progress),
        _after_install(drive_root, repo_dir))
    return JSONResponse({"ok": True, "job_id": job_id, "name": name, "basis": plan["basis"]})


async def api_skill_update(request: Request) -> JSONResponse:
    from ouroboros.skill_source_install import update_installed

    name = str(request.path_params.get("skill") or "").strip()
    body = await request_json_or(request, {}, exceptions=(Exception,))
    body = body if isinstance(body, dict) else {}
    mode = str(body.get("mode") or "")
    if mode not in {"replace", "keep", "merge"}:
        return json_error("mode is replace, keep or merge", 400)
    skip_review = body.get("skip_review", False)
    if not isinstance(skip_review, bool) or (skip_review and mode != "replace"):
        return json_error("skip_review is a boolean for mode replace only", 400)
    drive_root, repo_dir = request_drive_root(request), request_repo_dir(request)
    kwargs = {"mode": mode, "digest": str(body.get("digest") or ""),
              "local_digest": str(body.get("local_digest") or ""), "skip_review": skip_review,
              "actor": "owner", "repo_dir": repo_dir}
    if mode != "replace":
        try:
            result = await run_lifecycle_job(
                kind="update", target=name, source="skills", message=f"Updating {name} ({mode})",
                runner=lambda: run_blocking_preserving_cancellation(update_installed, drive_root, name, **kwargs))
        except SkillInstallError as exc:
            return _refusal(exc)
        return JSONResponse(result)
    try:
        await asyncio.to_thread(_precheck_update, drive_root, name, kwargs)
    except SkillInstallError as exc:
        return _refusal(exc)
    job_id = await _start_job("update", name, f"Updating {name}",
                              lambda progress: update_installed(drive_root, name, progress=progress, **kwargs),
                              _after_install(drive_root, repo_dir))
    return JSONResponse({"ok": True, "name": name, "mode": "replace", "job_id": job_id})


def _precheck_update(drive_root: pathlib.Path, name: str, kwargs: Dict[str, Any]) -> None:
    """Refuse a stale dialog before queueing (the lane re-checks everything)."""
    from ouroboros.config import get_skills_repo_path
    from ouroboros.skill_install_record import read_install_record
    from ouroboros.skill_loader import find_skill
    from ouroboros.skill_source_catalog import installed_release_digest
    from ouroboros.skill_source_install import _candidate_plan
    from ouroboros.skill_sources import load_sources

    record = read_install_record(drive_root, name)
    loaded = find_skill(drive_root, name, repo_path=get_skills_repo_path())
    if record is None or loaded is None:
        raise SkillInstallError("not_from_source", f"{name} was not installed from a source.", 404)
    if (record.get("source") or {}).get("kind") == "registry":
        return
    item = _candidate_plan(drive_root, name, record, load_sources(), kwargs.get("repo_dir"))["item"]
    if kwargs["digest"] and kwargs["digest"] != item["digest"]:
        raise SkillInstallError("candidate_changed", f"The source now offers a different release of {name}.",
                                candidate={"version": item.get("version", ""), "digest": item["digest"]})
    if kwargs["local_digest"] and kwargs["local_digest"] != installed_release_digest(loaded):
        raise SkillInstallError("local_changed", f"The local copy of {name} changed; look at it again.")


async def api_skill_uninstall(request: Request) -> JSONResponse:
    from ouroboros.skill_source_install import uninstall_installed

    name = str(request.path_params.get("skill") or "").strip()
    drive_root = request_drive_root(request)
    try:
        result = await run_lifecycle_job(
            kind="uninstall", target=name, source="skills", message=f"Uninstalling {name}",
            runner=lambda: run_blocking_preserving_cancellation(uninstall_installed, drive_root, name))
    except SkillInstallError as exc:
        return _refusal(exc)
    return JSONResponse(result)


async def api_onboarding_starter_skills(request: Request) -> JSONResponse:
    from ouroboros.skill_starter import starter_groups

    try:
        return JSONResponse(await asyncio.to_thread(starter_groups, request_drive_root(request),
                                                    repo_dir=request_repo_dir(request)))
    except Exception as exc:
        return json_exception(exc, context="starter skills read failed")


def skill_source_routes() -> list[Route]:
    return [
        Route("/api/skill-sources", endpoint=api_skill_sources, methods=["GET"]),
        Route("/api/skill-sources", endpoint=api_skill_source_add, methods=["POST"]),
        Route("/api/skill-sources/probe", endpoint=api_skill_source_probe, methods=["POST"]),
        Route("/api/skill-sources/{source_id}", endpoint=api_skill_source_change, methods=["POST"]),
        Route("/api/skill-sources/{source_id}/remove", endpoint=api_skill_source_remove, methods=["POST"]),
        Route("/api/skill-sources/{source_id}/refresh", endpoint=api_skill_source_refresh, methods=["POST"]),
        Route("/api/skill-catalog", endpoint=api_skill_catalog, methods=["GET"]),
        Route("/api/skill-catalog/search", endpoint=api_skill_catalog_search, methods=["GET"]),
        Route("/api/skill-catalog/item/{source_id}/{name}", endpoint=api_skill_catalog_item, methods=["GET"]),
        Route("/api/skill-catalog/install", endpoint=api_skill_catalog_install, methods=["POST"]),
        Route("/api/skills/{skill}/update", endpoint=api_skill_update, methods=["POST"]),
        Route("/api/skills/{skill}/uninstall", endpoint=api_skill_uninstall, methods=["POST"]),
        Route("/api/onboarding/starter-skills", endpoint=api_onboarding_starter_skills, methods=["GET"]),
    ]


__all__ = ["skill_source_routes"]

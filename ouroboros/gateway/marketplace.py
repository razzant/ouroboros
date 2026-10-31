"""ClawHub routes, the frozen hub-installed listing and the local publication receipt.

Skills from connected sources (the hub among them) install through ``gateway/skill_sources.py``;
these routes keep the ClawHub registry's own browse/install/update/uninstall surface. No route here
queues an automatic repair: fixing a skill is the owner's explicit "Ask Ouroboros to fix".
"""

from __future__ import annotations

import asyncio
import logging
import pathlib
from typing import Any, Dict, Optional

from starlette.requests import Request
from starlette.responses import JSONResponse

from ouroboros.marketplace.clawhub import (
    ClawHubClientError,
    ClawHubClientHostBlocked,
    ClawHubNotFoundError,
    ClawHubRateLimitError,
    info as _registry_info,
    search as _registry_search,
)
from ouroboros.marketplace.install import (
    install_skill,
    uninstall_skill,
    update_skill,
)
from ouroboros.marketplace.provenance import read_provenance
from ouroboros.skill_lifecycle_queue import (
    JobProgressTarget,
    LifecycleJobOptions,
    run_blocking_preserving_cancellation,
    run_lifecycle_job,
)

log = logging.getLogger(__name__)


from ouroboros.gateway._helpers import (
    coerce_bool as _coerce_bool,
    coerce_int as _coerce_int,
    json_error,
    json_exception,
    request_json_or,
    request_drive_root as _request_drive_root,
    request_repo_dir as _request_repo_dir,
)


def _client_error_response(exc: Exception, *, default_status: int = 502) -> JSONResponse:
    """Map a registry-client exception to a JSON error response.

    Upstream status is preserved where meaningful: a missing slug surfaces as
    404 (not a generic 502 bad-gateway), and rate limiting as 429.
    """
    if isinstance(exc, ClawHubClientHostBlocked):
        status = 400
    elif isinstance(exc, ClawHubNotFoundError):
        status = 404
    elif isinstance(exc, ClawHubRateLimitError):
        status = 429
    elif isinstance(exc, ClawHubClientError):
        status = default_status
    else:
        status = 500
    log.warning("marketplace error: %s", exc, exc_info=True)
    return JSONResponse({"error": str(exc), "code": exc.__class__.__name__}, status_code=status)


async def api_marketplace_search(request: Request) -> JSONResponse:
    qp = request.query_params
    query = qp.get("q") or qp.get("query") or ""
    sort = qp.get("sort") or "registry"
    limit = _coerce_int(qp.get("limit"), 25)
    include_plugins = _coerce_bool(qp.get("include_plugins"), False)
    official_only = _coerce_bool(qp.get("official") or qp.get("only_official"), False)
    cursor = qp.get("cursor") or None
    is_text_search = bool(str(query or "").strip())
    effective_cursor = None if is_text_search else cursor
    registry_official_only = False if is_text_search else official_only
    try:
        page = await asyncio.to_thread(
            _registry_search,
            query,
            limit=limit,
            sort=sort,
            cursor=effective_cursor,
            official_only=registry_official_only,
            include_metadata=True,
            timeout_sec=15 if is_text_search else 5,
        )
    except Exception as exc:
        return _client_error_response(exc)
    results = list(page.get("results") or [])
    if not include_plugins:
        results = [r for r in results if not r.is_plugin]
    if is_text_search and official_only:
        results = [r for r in results if bool((r.badges or {}).get("official"))]
    return JSONResponse(
        {
            "query": query,
            "sort": sort,
            "limit": limit,
            "offset": 0,
            "cursor": effective_cursor,
            "next_cursor": page.get("next_cursor") or "",
            "official": official_only,
            "registry_path": page.get("path") or "packages",
            "registry_attempts": page.get("attempts") or [],
            "registry_warnings": page.get("warnings") or [],
            "registry_empty": not bool(results),
            "count": len(results),
            "results": [r.to_dict() for r in results],
        }
    )


async def api_marketplace_info(request: Request) -> JSONResponse:
    slug = (request.path_params.get("slug") or "").strip()
    if not slug:
        return json_error("missing slug", 400)
    try:
        summary = await asyncio.to_thread(_registry_info, slug)
    except Exception as exc:
        return _client_error_response(exc)
    return JSONResponse(summary.to_dict())


def _preview_pipeline(slug: str, version: Optional[str]) -> Dict[str, Any]:
    """Return registry details for the lightweight ClawHub preview route.

    Preview is registry-metadata only: it does not download/stage/adapt the
    archive, so ``adapter.ok`` here reflects only the plugin check, not the
    archive-content validation (e.g. presence of ``SKILL.md``) that install
    enforces. ``metadata_only`` makes that explicit for the UI so an installable
    preview is not mistaken for an install-will-succeed guarantee.
    """
    summary = _registry_info(slug)
    return {
        "slug": slug,
        "version": version or summary.latest_version,
        "summary": summary.to_dict(),
        "metadata_only": True,
        "adapter": {
            "ok": not summary.is_plugin,
            "metadata_only": True,
            "warnings": (
                [] if summary.is_plugin else [
                    "Preview reflects registry metadata only; the archive is not "
                    "validated here. Final checks (SKILL.md presence, adapter "
                    "translation, review) run at install."
                ]
            ),
            "blockers": (
                ["OpenClaw Node/TypeScript plugins are not installable in Ouroboros."]
                if summary.is_plugin else []
            ),
        },
        "staging": {"is_plugin": summary.is_plugin},
    }


async def api_marketplace_preview(request: Request) -> JSONResponse:
    slug = (request.path_params.get("slug") or "").strip()
    if not slug:
        return json_error("missing slug", 400)
    version = (request.query_params.get("version") or "").strip() or None
    try:
        payload = await asyncio.to_thread(_preview_pipeline, slug, version)
    except ClawHubClientError as exc:
        return _client_error_response(exc)
    except Exception as exc:
        return json_exception(exc, context="marketplace preview failed")
    return JSONResponse(payload)

def _serialize_install_result(result: Any) -> Dict[str, Any]:
    """Project an :class:`InstallResult` into a JSON-friendly dict."""
    payload: Dict[str, Any] = {
        "ok": bool(result.ok),
        "sanitized_name": result.sanitized_name,
        "error": result.error,
    }
    if result.target_dir is not None:
        payload["target_dir"] = str(result.target_dir)
    if result.summary is not None:
        payload["summary"] = result.summary.to_dict()
    if result.archive is not None:
        payload["archive"] = {
            "sha256": result.archive.sha256,
            "size_bytes": len(result.archive.content),
            "version": result.archive.version,
        }
    if result.adapter is not None:
        payload["adapter"] = {
            "ok": result.adapter.ok,
            "warnings": result.adapter.warnings,
            "blockers": result.adapter.blockers,
            "sanitized_name": result.adapter.sanitized_name,
            "is_plugin": result.adapter.is_plugin,
        }
    payload["review_status"] = result.review_status
    payload["review_findings"] = result.review_findings
    payload["review_error"] = result.review_error
    payload["deps_status"] = getattr(result, "deps_status", "")
    payload["deps_error"] = getattr(result, "deps_error", "")
    payload["deps_fingerprint"] = getattr(result, "deps_fingerprint", {})
    payload["provenance"] = result.provenance
    return payload


def _lifecycle_options(
    success_prefix: str,
    failure_fallback: str,
    *,
    progress_target: JobProgressTarget | None = None,
    object_result: bool = False,
    include_deps_error: bool = False,
) -> LifecycleJobOptions:
    get_value = (lambda item, key, default=None: getattr(item, key, default) if object_result else item.get(key, default))

    return LifecycleJobOptions(
        progress_target=progress_target,
        result_message=lambda item: (
            f"{success_prefix} {get_value(item, 'sanitized_name')}"
            if get_value(item, "ok", False)
            else get_value(item, "error", failure_fallback)
        ),
        result_error=lambda item: (
            get_value(item, "error", failure_fallback)
            if not get_value(item, "ok", False)
            else (
                get_value(item, "deps_error", "")
                if include_deps_error and get_value(item, "deps_status", "") == "failed"
                else ""
            )
        ),
    )


async def api_marketplace_install(request: Request) -> JSONResponse:
    body = await request_json_or(request, {}, exceptions=(Exception,))
    body = body if isinstance(body, dict) else {}
    slug = str(body.get("slug") or "").strip()
    if not slug:
        return json_error("missing slug", 400)
    version = str(body.get("version") or "").strip() or None
    auto_review = _coerce_bool(body.get("auto_review"), True)
    overwrite = _coerce_bool(body.get("overwrite"), False)

    drive_root = _request_drive_root(request)
    repo_dir = _request_repo_dir(request)
    install_progress = JobProgressTarget()
    async def _run_install() -> Any:
        return await run_blocking_preserving_cancellation(
            install_skill,
            drive_root,
            repo_dir,
            slug=slug,
            version=version,
            auto_review=auto_review,
            overwrite=overwrite,
            progress_callback=install_progress.set,
            log_label="ClawHub install lifecycle operation",
        )

    try:
        result = await run_lifecycle_job(
            kind="install",
            target=slug,
            source="clawhub",
            message=f"Installing {slug}",
            runner=_run_install,
            options=_lifecycle_options(
                "Installed as",
                "install failed",
                progress_target=install_progress,
                object_result=True,
                include_deps_error=True,
            ),
        )
    except Exception as exc:
        return json_exception(exc, context="marketplace install failed")
    payload = _serialize_install_result(result)
    status = 200 if result.ok else (getattr(result, "error_status", 0) or 400)
    return JSONResponse(payload, status_code=status)


async def api_marketplace_update(request: Request) -> JSONResponse:
    sanitized = (request.path_params.get("name") or "").strip()
    err = _validate_path_param_name(sanitized)
    if err:
        return json_error(err, 400)
    body = await request_json_or(request, {}, exceptions=(Exception,))
    body = body if isinstance(body, dict) else {}
    version = str(body.get("version") or "").strip() or None
    drive_root = _request_drive_root(request)
    repo_dir = _request_repo_dir(request)
    update_progress = JobProgressTarget()
    async def _run_update() -> Any:
        return await run_blocking_preserving_cancellation(
            update_skill,
            drive_root,
            repo_dir,
            sanitized_name=sanitized,
            version=version,
            progress_callback=update_progress.set,
            log_label="ClawHub update lifecycle operation",
        )

    try:
        result = await run_lifecycle_job(
            kind="update",
            target=sanitized,
            source="clawhub",
            message=f"Updating {sanitized}",
            runner=_run_update,
            options=_lifecycle_options(
                "Updated",
                "update failed",
                progress_target=update_progress,
                object_result=True,
                include_deps_error=True,
            ),
        )
    except Exception as exc:
        return json_exception(exc, context="marketplace update failed")
    payload = _serialize_install_result(result)
    status = 200 if result.ok else (getattr(result, "error_status", 0) or 400)
    return JSONResponse(payload, status_code=status)


def _validate_path_param_name(name: str) -> Optional[str]:
    """Reject traversal-like names at the HTTP boundary; downstream revalidates."""
    cleaned = (name or "").strip()
    if not cleaned:
        return "missing name"
    if cleaned in {".", ".."}:
        return f"invalid name: {cleaned!r}"
    if "/" in cleaned or "\\" in cleaned or "\x00" in cleaned:
        return f"name must not contain path separators: {cleaned!r}"
    return None


def _installed_skill_payload(skill: Any, drive_root: pathlib.Path, *, provenance: Dict[str, Any] | None = None) -> Dict[str, Any]:
    from ouroboros.skill_loader import grant_status_for_skill

    try:
        rel_skill_dir = skill.skill_dir.resolve().relative_to(drive_root.resolve())
        payload_root = rel_skill_dir.as_posix() if rel_skill_dir.parts[:1] == ("skills",) else ""
    except Exception:
        payload_root = ""
    stale = skill.review.is_stale_for(skill.content_hash)
    gate = skill.review.gate_for(skill.content_hash)
    payload = {
        "name": skill.name,
        "type": skill.manifest.type,
        "version": skill.manifest.version,
        "review_status": skill.review.status,
        "review_stale": stale,
        "review_gate": gate,
        "executable_review": gate["executable_review"],
        "review_findings": list(skill.review.findings or []),
        "enabled": skill.enabled,
        "load_error": skill.load_error,
        "grants": grant_status_for_skill(drive_root, skill),
        "payload_root": payload_root,
    }
    if provenance is not None:
        payload["provenance"] = provenance
    return payload


def _installed_skills_for_source(drive_root: pathlib.Path, source: str) -> list[Dict[str, Any]]:
    from ouroboros.config import get_skills_repo_path
    from ouroboros.skill_install_record import read_install_record
    from ouroboros.skill_loader import discover_skills

    out: list[Dict[str, Any]] = []
    for skill in discover_skills(drive_root, repo_path=get_skills_repo_path()):
        hub_record = None
        if source == "ouroboroshub" and not skill.identity_collision:
            hub_record = read_install_record(drive_root, skill.name)
        if source == "ouroboroshub":
            from_hub = (((hub_record or {}).get("source") or {}).get("kind") == "hub" or
                        (hub_record is None and skill.location == source))
            if not from_hub:
                continue
        elif skill.source != source:
            continue
        provenance = None
        if source == "clawhub":
            provenance = {}
            if not bool(getattr(skill, "identity_collision", False)):
                provenance = read_provenance(drive_root, skill.name) or {}
        out.append(_installed_skill_payload(skill, drive_root, provenance=provenance))
    return out


async def api_marketplace_uninstall(request: Request) -> JSONResponse:
    sanitized = (request.path_params.get("name") or "").strip()
    err = _validate_path_param_name(sanitized)
    if err:
        return json_error(err, 400)
    drive_root = _request_drive_root(request)
    async def _run_uninstall() -> Any:
        return await run_blocking_preserving_cancellation(
            uninstall_skill,
            drive_root,
            sanitized_name=sanitized,
            log_label="ClawHub uninstall lifecycle operation",
        )

    try:
        result = await run_lifecycle_job(
            kind="uninstall",
            target=sanitized,
            source="clawhub",
            message=f"Uninstalling {sanitized}",
            runner=_run_uninstall,
            options=_lifecycle_options("Uninstalled", "uninstall failed", object_result=True),
        )
    except Exception as exc:
        return json_exception(exc, context="marketplace uninstall failed")
    return JSONResponse(
        {
            "ok": result.ok,
            "sanitized_name": result.sanitized_name,
            "error": result.error,
        },
        status_code=200 if result.ok else 400,
    )


async def api_marketplace_installed(request: Request) -> JSONResponse:
    """List ClawHub-installed skills + provenance for the UI."""
    drive_root = _request_drive_root(request)
    out = _installed_skills_for_source(drive_root, "clawhub")
    return JSONResponse({"count": len(out), "skills": out})


async def api_hub_installed(request: Request) -> JSONResponse:
    """Frozen compatibility list: hub-source installs and the legacy hub bucket, in the registry shape."""
    out = await asyncio.to_thread(_installed_skills_for_source, _request_drive_root(request), "ouroboroshub")
    return JSONResponse({"count": len(out), "skills": out})

async def api_skill_publication_clear(request: Request) -> JSONResponse:
    """Forget the displayed publication receipt through its existing local receipt owner."""
    from ouroboros.marketplace.provenance import PublicationRecordChanged, clear_publication_record

    name = str(request.path_params.get("skill") or "").strip()
    if error := _validate_path_param_name(name):
        return json_error(error, 400)
    body = await request_json_or(request, {}, exceptions=(Exception,))
    expected = body.get("expected_published") if isinstance(body, dict) else None
    if not isinstance(expected, dict):
        return json_error("expected_published must contain the displayed publication receipt", 400)
    try:
        cleared = await asyncio.to_thread(clear_publication_record, _request_drive_root(request), name, expected)
    except PublicationRecordChanged as exc:
        return JSONResponse({"ok": False, "code": "publication_changed", "error": str(exc)}, status_code=409)
    except ValueError as exc:
        return JSONResponse({"ok": False, "code": "publication_invalid", "error": str(exc)}, status_code=400)
    except Exception as exc:
        return json_exception(exc, context="Clearing local publication failed")
    return JSONResponse({"ok": True, "sanitized_name": name, "publication_cleared": cleared})


__all__ = [
    "api_marketplace_search",
    "api_marketplace_info",
    "api_marketplace_preview",
    "api_marketplace_install",
    "api_marketplace_update",
    "api_marketplace_uninstall",
    "api_marketplace_installed",
    "api_hub_installed",
    "api_skill_publication_clear",
]

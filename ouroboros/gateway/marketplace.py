"""HTTP surface for ClawHub/OuroborosHub marketplace routes."""

from __future__ import annotations

import asyncio
import json
import logging
import pathlib
import re
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
    _run_skill_review,
    discard_payload_snapshot,
    install_skill,
    restore_payload_state,
    snapshot_payload_state,
    uninstall_skill,
    update_skill,
)
from ouroboros.marketplace.provenance import read_provenance
from ouroboros.marketplace import ouroboroshub
from ouroboros.skill_lifecycle_queue import (
    JobProgressTarget,
    LifecycleJobOptions,
    run_blocking_preserving_cancellation,
    run_lifecycle_job,
)
from ouroboros.utils import atomic_write_json, read_json_dict, utc_now_iso

log = logging.getLogger(__name__)


def _reconcile_deps_after_review(drive_root: pathlib.Path, skill_name: str) -> tuple[str, str]:
    from ouroboros.skill_review_runner import _reconcile_deps_after_pass_review

    return _reconcile_deps_after_pass_review(drive_root, skill_name)


def _review_status_allows_skill_runtime(status: str) -> bool:
    from ouroboros.skill_loader import review_status_allows_execution

    return review_status_allows_execution(status)


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


# Serialization lives beside the transaction bodies in the marketplace layer.
_serialize_hub_install_result = ouroboroshub.serialize_hub_install_result

# Typed hub error codes carry their own HTTP status (§7.3); anything untyped
# keeps the historical 200-or-400 contract.
_HUB_ERROR_HTTP_STATUS = {
    "adopt_cas_mismatch": 409,
    "adopt_not_eligible": 409,
    "catalog_identity_conflict": 409,
    "catalog_unavailable": 502,
}


def _hub_payload_status(payload: Dict[str, Any]) -> int:
    if payload.get("ok"):
        return 200
    return _HUB_ERROR_HTTP_STATUS.get(str(payload.get("code") or ""), 400)


async def _apply_hub_review_and_deps(
    payload: Dict[str, Any],
    *,
    drive_root: pathlib.Path,
    repo_dir: pathlib.Path,
    skill_name: str,
    progress: JobProgressTarget,
    review_log_label: str,
    deps_log_label: str,
) -> tuple[str, str, str]:
    progress.set("Running tri-model review…")
    status, findings, error = await run_blocking_preserving_cancellation(
        _run_skill_review,
        drive_root,
        repo_dir,
        skill_name,
        log_label=review_log_label,
    )
    payload.update({"review_status": status, "review_findings": findings, "review_error": error})
    deps_status = "not_required"
    deps_error = ""
    if _review_status_allows_skill_runtime(status) and not error:
        progress.set("Installing dependencies…")
        deps_status, deps_error = await run_blocking_preserving_cancellation(
            _reconcile_deps_after_review,
            drive_root,
            skill_name,
            log_label=deps_log_label,
        )
        payload.update({"deps_status": deps_status, "deps_error": deps_error})
        if deps_status == "failed":
            payload["ok"] = False
            payload["error"] = deps_error
    return status, error, deps_status


def _resync_skill_schedules_quiet(drive_root: pathlib.Path) -> None:
    """Mirror skill manifest schedules after a marketplace lifecycle change so a
    removed/renamed/updated scheduled skill does not fire stale before the
    periodic scheduler tick. Handlers call it through ``asyncio.to_thread``:
    the mirror waits for the supervisor queue lock, and that wait must stay off
    the HTTP event loop (the scheduler tick already runs the same idempotent
    resync beside lifecycle operations, so nothing new interleaves)."""
    try:
        from supervisor.queue import resync_skill_schedules

        resync_skill_schedules(drive_root)
    except Exception:
        log.debug("marketplace schedule resync failed", exc_info=True)


def _auto_repair_marker_path(drive_root: pathlib.Path, skill_name: str) -> pathlib.Path:
    from ouroboros.skill_loader import skill_state_dir

    return skill_state_dir(drive_root, skill_name) / "auto_repair.json"


def _maybe_enqueue_marketplace_auto_repair(
    drive_root: pathlib.Path,
    *,
    skill_name: str,
    source: str,
    reason: str,
    review_findings: list[Dict[str, Any]] | None = None,
) -> bool:
    """Queue one review-mediated repair task per marketplace payload hash."""

    try:
        from ouroboros.config import get_skills_repo_path
        from ouroboros.skill_loader import find_skill
        from supervisor.message_bus import get_bridge

        skill = find_skill(drive_root, skill_name, repo_path=get_skills_repo_path())
        if skill is None:
            return False
        content_hash = str(skill.content_hash or "").strip()
        if not content_hash:
            return False
        marker_path = _auto_repair_marker_path(drive_root, skill.name)
        marker = read_json_dict(marker_path) if marker_path.is_file() else {}
        attempted = set(str(item) for item in (marker.get("attempted_hashes") or []))
        if content_hash in attempted:
            return False
        try:
            payload_root = skill.skill_dir.resolve().relative_to(pathlib.Path(drive_root).resolve()).as_posix()
        except Exception:
            return False
        findings = list(review_findings or [])[:12]
        prompt = (
            "Repair the marketplace-installed skill payload.\n\n"
            f"Source: {source}\n"
            f"Skill: {skill.name}\n"
            f"Payload root: {payload_root}\n"
            f"Reason: {reason}\n"
            f"Content hash: {content_hash}\n\n"
            "Constraints:\n"
            "- Edit only this skill payload.\n"
            "- Do not enable the skill or grant secrets/permissions.\n"
            "- Preserve marketplace provenance and dependency markers.\n"
            "- Re-run skill review after edits and stop if review still blocks execution.\n\n"
            "Review findings:\n"
            + json.dumps(findings, ensure_ascii=False, indent=2)
        )
        bridge = get_bridge()
        bridge.ui_send(
            prompt,
            broadcast=False,
            suppress_chat_log=True,
            task_constraint={
                "mode": "normal",
                "skill_name": skill.name,
                "payload_root": payload_root,
                "allow_enable": False,
                "allow_review": True,
            },
        )
        attempted.add(content_hash)
        atomic_write_json(
            marker_path,
            {
                "schema_version": 1,
                "skill": skill.name,
                "source": source,
                "attempted_hashes": sorted(attempted),
                "last_attempted_hash": content_hash,
                "last_enqueued_at": utc_now_iso(),
            },
            trailing_newline=True,
        )
        # X3: no invented task ids. `ui_send` enqueues a message the router
        # promotes into a managed task LATER — no task id exists yet here, and
        # the old synthetic `skill_repair_<name>_<hash8>` string looked exactly
        # like one. The receipt says the truth: enqueued, id pending.
        bridge.broadcast({
            "type": "chat",
            "role": "system",
            "content": (
                f"Repair task queued for {skill.name} (task id pending promotion). "
                "Ouroboros will inspect the skill payload and re-run review."
            ),
            "ts": utc_now_iso(),
            "source": "skill_repair",
            "system_type": "skill_repair",
            "task_id": "",
            "task_id_pending": True,
        })
        return True
    except Exception:
        log.debug("marketplace auto-repair enqueue failed for %s", skill_name, exc_info=True)
        return False


def _maybe_enqueue_repair_for_payload(drive_root: pathlib.Path, payload: Dict[str, Any], *, source: str) -> None:
    if str(payload.get("review_status") or "") != "blockers":
        return
    if payload.get("rolled_back"):
        # A rolled-back transaction restored the PREVIOUS payload (for adopt:
        # the owner's own external copy). The blockers describe the discarded
        # replacement tree, so a repair task aimed at the restored payload
        # would "fix" files the findings never reviewed.
        return
    skill_name = str(payload.get("sanitized_name") or "").strip()
    if not skill_name:
        return
    _maybe_enqueue_marketplace_auto_repair(
        drive_root,
        skill_name=skill_name,
        source=source,
        reason="marketplace review blockers",
        review_findings=list(payload.get("review_findings") or []),
    )


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
    # Resync regardless of ok: a deps-failure can set ok=false after the payload
    # was already installed on disk, changing scheduled-task readiness.
    await asyncio.to_thread(_resync_skill_schedules_quiet, drive_root)
    payload = _serialize_install_result(result)
    _maybe_enqueue_repair_for_payload(drive_root, payload, source="clawhub")
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
    # Resync regardless of ok: an update can mutate the payload on disk even when
    # a follow-up deps step reports ok=false, changing scheduled-task readiness.
    await asyncio.to_thread(_resync_skill_schedules_quiet, drive_root)
    payload = _serialize_install_result(result)
    _maybe_enqueue_repair_for_payload(drive_root, payload, source="clawhub")
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
    from ouroboros.skill_loader import discover_skills

    out: list[Dict[str, Any]] = []
    for skill in discover_skills(drive_root, repo_path=get_skills_repo_path()):
        if skill.source != source:
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
    if result.ok:
        # The skill is gone; drop its scheduled tasks now so the scheduler does
        # not fire a deleted skill before the next periodic resync.
        await asyncio.to_thread(_resync_skill_schedules_quiet, drive_root)
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

async def api_ouroboroshub_catalog(request: Request) -> JSONResponse:
    query = str(request.query_params.get("q") or request.query_params.get("query") or "").strip()
    try:
        # Display read: the only consumer of the §7.1a catalog TTL memo.
        results = await asyncio.to_thread(ouroboroshub.search, query, fresh=False)
    except Exception as exc:
        log.warning("OuroborosHub catalog failed: %s", exc, exc_info=True)
        return json_exception(exc, 502)
    return JSONResponse({"query": query, "count": len(results), "results": [item.to_dict() for item in results]})


async def api_ouroboroshub_preview(request: Request) -> JSONResponse:
    slug = str(request.path_params.get("slug") or "").strip()
    if not slug:
        return json_error("missing slug", 400)
    try:
        summary = await asyncio.to_thread(ouroboroshub.info, slug)
    except Exception as exc:
        return json_exception(exc, 404)
    return JSONResponse({"summary": summary.to_dict(), "files": summary.files})


_ADOPT_HASH_RE = re.compile(r"^[0-9a-f]{64}$")


def _validate_adopt_body(body: Dict[str, Any], sanitized: str) -> Optional[JSONResponse]:
    """§7.3 body validation for adopt=true; None means the body is acceptable."""
    def _reject(error: str, code: str) -> JSONResponse:
        return JSONResponse(
            {"ok": False, "sanitized_name": sanitized, "error": error, "code": code},
            status_code=400,
        )

    expected = str(body.get("expected_content_hash") or "").strip()
    if not expected:
        return _reject("adopt requires expected_content_hash", "adopt_expected_hash_missing")
    if not _ADOPT_HASH_RE.fullmatch(expected):
        return _reject(
            "expected_content_hash must be 64 lowercase hex characters",
            "adopt_expected_hash_invalid",
        )
    if "auto_review" in body and not _coerce_bool(body.get("auto_review"), True):
        return _reject("adopt always runs auto review", "adopt_requires_auto_review")
    if _coerce_bool(body.get("overwrite"), False):
        return _reject("adopt and overwrite are mutually exclusive", "adopt_overwrite_conflict")
    return None


async def _api_ouroboroshub_adopt(request: Request, body: Dict[str, Any], slug: str) -> JSONResponse:
    """Adopt dispatch (§7.3): validate the body, SKIP the pre-lifecycle identity
    precheck (the transaction's move-aside makes the installer's own precheck
    pass naturally), and run the transaction through the lifecycle machinery."""
    from ouroboros.skill_loader import _sanitize_skill_name

    sanitized = _sanitize_skill_name(slug)
    refusal = _validate_adopt_body(body, sanitized)
    if refusal is not None:
        return refusal
    expected = str(body.get("expected_content_hash") or "").strip()
    drive_root = _request_drive_root(request)
    repo_dir = _request_repo_dir(request)
    adopt_progress = JobProgressTarget()

    async def _apply_review_and_deps(payload: Dict[str, Any], skill_name: str) -> tuple[str, str, str]:
        return await _apply_hub_review_and_deps(
            payload,
            drive_root=drive_root,
            repo_dir=repo_dir,
            skill_name=skill_name,
            progress=adopt_progress,
            review_log_label="OuroborosHub adopt review lifecycle operation",
            deps_log_label="OuroborosHub adopt dependency lifecycle operation",
        )

    async def _run_adopt() -> Dict[str, Any]:
        return await ouroboroshub.run_hub_adopt(
            slug,
            drive_root=drive_root,
            expected_content_hash=expected,
            progress=adopt_progress,
            run_blocking=run_blocking_preserving_cancellation,
            apply_review_and_deps=_apply_review_and_deps,
        )

    payload = await run_lifecycle_job(
        kind="install",
        target=slug,
        source="ouroboroshub",
        message=f"Adopting {slug}",
        runner=_run_adopt,
        options=_lifecycle_options(
            "Adopted",
            "adopt failed",
            progress_target=adopt_progress,
        ),
    )
    # Resync regardless of ok: a rolled-back adopt still restored payloads on
    # disk and a successful one changed the scheduled-task inventory.
    await asyncio.to_thread(_resync_skill_schedules_quiet, drive_root)
    _maybe_enqueue_repair_for_payload(drive_root, payload, source="ouroboroshub")
    return JSONResponse(payload, status_code=_hub_payload_status(payload))


async def api_ouroboroshub_install(request: Request) -> JSONResponse:
    body = await request_json_or(request, {}, exceptions=(Exception,))
    body = body if isinstance(body, dict) else {}
    slug = str(body.get("slug") or "").strip()
    if not slug:
        return json_error("missing slug", 400)
    if _coerce_bool(body.get("adopt"), False):
        return await _api_ouroboroshub_adopt(request, body, slug)
    overwrite = _coerce_bool(body.get("overwrite"), False)
    auto_review = _coerce_bool(body.get("auto_review"), True)
    drive_root = _request_drive_root(request)
    repo_dir = _request_repo_dir(request)
    from ouroboros.skill_loader import _sanitize_skill_name

    sanitized = _sanitize_skill_name(slug)
    identity_error = await asyncio.to_thread(
        ouroboroshub.install_identity_error,
        sanitized,
        drive_root=drive_root,
    )
    if identity_error:
        return json_error(identity_error, 409)
    install_progress = JobProgressTarget()

    async def _run_install() -> Dict[str, Any]:
        target_dir = drive_root / "skills" / "ouroboroshub" / sanitized
        rollback_snapshot = snapshot_payload_state(drive_root, sanitized, target_dir)
        install_progress.set("Downloading from OuroborosHub…")
        try:
            result = await run_blocking_preserving_cancellation(
                ouroboroshub.install,
                slug,
                overwrite=overwrite,
                log_label="OuroborosHub install lifecycle operation",
            )
            payload = _serialize_hub_install_result(result)
            deps_status = ""
            if result.ok and auto_review:
                _status, _error, deps_status = await _apply_hub_review_and_deps(
                    payload,
                    drive_root=drive_root,
                    repo_dir=repo_dir,
                    skill_name=result.sanitized_name,
                    progress=install_progress,
                    review_log_label="OuroborosHub install review lifecycle operation",
                    deps_log_label="OuroborosHub install dependency lifecycle operation",
                )
            if result.ok and deps_status == "failed":
                restore_payload_state(rollback_snapshot)
            else:
                discard_payload_snapshot(rollback_snapshot)
            return payload
        except Exception:
            restore_payload_state(rollback_snapshot)
            raise

    payload = await run_lifecycle_job(
        kind="install",
        target=slug,
        source="ouroboroshub",
        message=f"Installing {slug}",
        runner=_run_install,
        options=_lifecycle_options(
            "Installed as",
            "install failed",
            progress_target=install_progress,
        ),
    )
    # Resync regardless of ok: install + deps can leave the payload on disk with
    # ok=false, changing scheduled-task readiness.
    await asyncio.to_thread(_resync_skill_schedules_quiet, drive_root)
    _maybe_enqueue_repair_for_payload(drive_root, payload, source="ouroboroshub")
    return JSONResponse(payload, status_code=_hub_payload_status(payload))


async def api_ouroboroshub_update(request: Request) -> JSONResponse:
    name = str(request.path_params.get("name") or "").strip()
    err = _validate_path_param_name(name)
    if err:
        return json_error(err, 400)
    drive_root = _request_drive_root(request)
    repo_dir = _request_repo_dir(request)
    identity_error = await asyncio.to_thread(
        ouroboroshub.install_identity_error,
        name,
        drive_root=drive_root,
    )
    if identity_error:
        return json_error(identity_error, 409)
    update_progress = JobProgressTarget()

    async def _apply_review_and_deps(payload: Dict[str, Any], skill_name: str) -> tuple[str, str, str]:
        return await _apply_hub_review_and_deps(
            payload,
            drive_root=drive_root,
            repo_dir=repo_dir,
            skill_name=skill_name,
            progress=update_progress,
            review_log_label="OuroborosHub update review lifecycle operation",
            deps_log_label="OuroborosHub update dependency lifecycle operation",
        )

    async def _run_update() -> Dict[str, Any]:
        return await ouroboroshub.run_hub_update(
            name,
            drive_root=drive_root,
            progress=update_progress,
            run_blocking=run_blocking_preserving_cancellation,
            apply_review_and_deps=_apply_review_and_deps,
        )

    payload = await run_lifecycle_job(
        kind="update",
        target=name,
        source="ouroboroshub",
        message=f"Updating {name}",
        runner=_run_update,
        options=_lifecycle_options(
            "Updated",
            "update failed",
            progress_target=update_progress,
        ),
    )
    # Resync regardless of ok: an update can mutate the payload on disk even when
    # a follow-up deps step reports ok=false, changing scheduled-task readiness.
    await asyncio.to_thread(_resync_skill_schedules_quiet, drive_root)
    _maybe_enqueue_repair_for_payload(drive_root, payload, source="ouroboroshub")
    return JSONResponse(payload, status_code=_hub_payload_status(payload))

async def api_ouroboroshub_installed(request: Request) -> JSONResponse:
    drive_root = _request_drive_root(request)
    out = _installed_skills_for_source(drive_root, "ouroboroshub")
    return JSONResponse({"count": len(out), "skills": out})


async def api_ouroboroshub_uninstall(request: Request) -> JSONResponse:
    sanitized = str(request.path_params.get("name") or "").strip()
    err = _validate_path_param_name(sanitized)
    if err:
        return json_error(err, 400)
    drive_root = _request_drive_root(request)

    async def _run_uninstall() -> Dict[str, Any]:
        result = await run_blocking_preserving_cancellation(
            ouroboroshub.uninstall,
            sanitized,
            log_label="OuroborosHub uninstall lifecycle operation",
        )
        return {"ok": result.ok, "sanitized_name": result.sanitized_name, "error": result.error}

    payload = await run_lifecycle_job(
        kind="uninstall",
        target=sanitized,
        source="ouroboroshub",
        message=f"Uninstalling {sanitized}",
        runner=_run_uninstall,
        options=_lifecycle_options("Uninstalled", "uninstall failed"),
    )
    if payload.get("ok"):
        await asyncio.to_thread(_resync_skill_schedules_quiet, drive_root)
    return JSONResponse(payload, status_code=200 if payload.get("ok") else 400)


async def api_ouroboroshub_clear_publication(request: Request) -> JSONResponse:
    """Clear the displayed submission through its existing local receipt owner."""
    from ouroboros.marketplace.provenance import PublicationRecordChanged, clear_publication_record

    name = str(request.path_params.get("name") or "").strip()
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
    "api_ouroboroshub_catalog",
    "api_ouroboroshub_preview",
    "api_ouroboroshub_install",
    "api_ouroboroshub_update",
    "api_ouroboroshub_installed",
    "api_ouroboroshub_uninstall",
    "api_ouroboroshub_clear_publication",
]

"""HTTP endpoints for extension catalogue, manifests, modules, and dispatch."""

from __future__ import annotations

import asyncio
import base64
import inspect
import logging
import pathlib
import shutil
import time
from datetime import datetime, timezone
from typing import Any, Dict

from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from ouroboros.extension_loader import list_routes, snapshot
from ouroboros.gateway._helpers import (
    coerce_bool,
    json_error,
    json_exception,
    request_drive_root as _request_drive_root,
    request_json_or,
    request_repo_dir as _request_repo_dir,
)
from ouroboros.skill_lifecycle_queue import (
    LifecycleJobOptions,
    queue_snapshot,
    run_blocking_preserving_cancellation,
    run_lifecycle_job,
)
from ouroboros.skill_loader import (
    discover_skills,
    find_skill,
    grant_status_for_skill,
    requested_core_setting_keys,
    requested_skill_permissions,
    review_status_allows_execution,
    save_skill_grants,
    skill_conflict_status,
    skill_review_gate,
    _sanitize_skill_name,
)
from ouroboros.skill_review_usage import (
    skill_review_attempt_coverage,
    skill_review_usage_markdown,
)
from ouroboros.utils import append_jsonl, utc_now_iso

log = logging.getLogger(__name__)
_CHILD_DISPATCH_HEADER_DENYLIST = {
    "authorization",
    "cookie",
    "proxy-authorization",
    "set-cookie",
    "x-api-key",
    "x-auth-token",
}
_CHILD_DISPATCH_BODY_CAP = 512 * 1024
# (name, content_hash) -> (verdict, monotonic stamp). TTL-bounded: the verdict
# also depends on the LIVE catalog, so an unexpiring memo could keep claiming
# "published" after the hub advanced or dropped the slug (final-gate finding).
_OFFICIAL_HUB_VERIFIED_HINT_CACHE: dict[tuple[str, str], tuple[bool, float]] = {}
_OFFICIAL_HUB_VERIFIED_TTL_SEC = 300.0


def _passive_submit_hub(
    loaded: Any,
    *,
    github_token_configured: bool | None = None,
    review_stale: bool | None = None,
) -> dict[str, Any]:
    """Project passive visibility/admission without running the scanner."""
    from ouroboros.skill_publish_eligibility import (
        PUBLISHABLE_SOURCES,
        submit_hub_eligibility,
    )

    if bool(getattr(loaded, "identity_collision", False)):
        return {
            "visible": False,
            "publication_ready": False,
            "task_start_allowed": False,
            "disabled": True,
            "state": "hard_block",
            "reason": "",
        }
    source = str(getattr(loaded, "source", "") or "")
    if source.lower() in PUBLISHABLE_SOURCES and github_token_configured is None:
        from ouroboros.tools.github import github_token_from_env_or_settings

        github_token_configured = bool(github_token_from_env_or_settings())
    return submit_hub_eligibility(
        source=source,
        review_status=loaded.review.status,
        review_profile=getattr(loaded.review, "review_profile", "") or "",
        review_stale=(
            loaded.review.is_stale_for(loaded.content_hash)
            if review_stale is None
            else bool(review_stale)
        ),
        github_token_configured=bool(github_token_configured),
    )


async def _read_child_dispatch_body(request: Request) -> bytes:
    raw_length = request.headers.get("content-length")
    if raw_length:
        try:
            if int(raw_length) > _CHILD_DISPATCH_BODY_CAP:
                raise ValueError("extension route body too large")
        except ValueError:
            raise ValueError("extension route body too large")
    chunks = bytearray()
    async for chunk in request.stream():
        if len(chunks) + len(chunk) > _CHILD_DISPATCH_BODY_CAP:
            raise ValueError("extension route body too large")
        chunks.extend(chunk)
    return bytes(chunks)


def _review_fields(
    loaded: Any, *, stale: bool | None = None, gate: dict[str, Any] | None = None,
    github_token_configured: bool | None = None,
) -> dict[str, Any]:
    stale = loaded.review.is_stale_for(loaded.content_hash) if stale is None else stale
    gate = (skill_review_gate(loaded.review.status, stale=stale,
                              findings=getattr(loaded.review, "findings", None))
            if gate is None else gate)
    source = str(getattr(loaded, "source", "") or "")
    official_hub_verified = False
    if source == "ouroboroshub":
        try:
            key = (str(getattr(loaded, "name", "") or ""), str(getattr(loaded, "content_hash", "") or ""))
            cached = _OFFICIAL_HUB_VERIFIED_HINT_CACHE.get(key) if key[0] and key[1] else None
            now = time.monotonic()
            if cached is not None and (now - cached[1]) < _OFFICIAL_HUB_VERIFIED_TTL_SEC:
                official_hub_verified = cached[0]
            else:
                from ouroboros.skill_review import is_official_hub_payload_verified

                official_hub_verified = bool(is_official_hub_payload_verified(loaded))
                if key[0] and key[1]:
                    # Evict expired entries so superseded content hashes do
                    # not accumulate across skill revisions.
                    for stale_key in [
                        k for k, (_, at) in _OFFICIAL_HUB_VERIFIED_HINT_CACHE.items()
                        if (now - at) >= _OFFICIAL_HUB_VERIFIED_TTL_SEC
                    ]:
                        _OFFICIAL_HUB_VERIFIED_HINT_CACHE.pop(stale_key, None)
                    _OFFICIAL_HUB_VERIFIED_HINT_CACHE[key] = (official_hub_verified, now)
        except Exception:
            official_hub_verified = False
    owner_attestable = (
        (source == "ouroboroshub" and official_hub_verified)
        or (source not in {"native", "clawhub", "ouroboroshub"} and (
            source == "external" or bool(getattr(loaded, "is_self_authored", False))
        ))
    )
    # FR1: the host computes the single Submit-to-Hub eligibility verdict so the card
    # renders it instead of recomputing a divergent clean-only rule (the SSOT shared with
    # the backend gate). The github-token check is request-INVARIANT, so the index builder
    # resolves it ONCE and threads it in; a single-skill caller (None) resolves it lazily
    # and only when the source is publishable — never a per-skill settings.json read on a
    # native-heavy GET /api/extensions.
    submit_hub = _passive_submit_hub(
        loaded,
        github_token_configured=github_token_configured,
        review_stale=stale,
    )
    return {
        "review_status": loaded.review.status,
        "review_stale": stale,
        "review_gate": gate,
        "executable_review": gate["executable_review"],
        # Surfaced so the UI can mark an owner-attested skill (LLM review skipped) distinctly
        # from a normal LLM-clean verdict, and hide the "Skip review" action once attested.
        "review_profile": getattr(loaded.review, "review_profile", ""),
        # UI hint only: the owner-attestation endpoint repeats the authoritative checks.
        "official_hub_verified": official_hub_verified,
        "owner_attestable": owner_attestable,
        # FR1: SSOT publish-eligibility verdict {visible, disabled, reason}.
        "submit_hub": submit_hub,
    }


def _broadcast_extension_lifecycle(request: Request, skill: str, action: Any, reason: Any = "") -> None:
    if not action:
        return
    try:
        broadcaster = getattr(request.app.state, "broadcast_ws_sync", None)
    except Exception:
        broadcaster = None
    if not callable(broadcaster):
        return
    broadcaster({
        "type": "extension_lifecycle",
        "skill": str(skill or ""),
        "action": str(action or ""),
        "reason": str(reason or ""),
    })


def _owner_grant_audit(drive_root: pathlib.Path, request: Request, payload: Dict[str, Any]) -> None:
    try:
        client = getattr(request, "client", None)
        append_jsonl(
            pathlib.Path(drive_root) / "logs" / "events.jsonl",
            {
                "ts": utc_now_iso(),
                "type": "owner_api_action",
                "action": "skill_grant",
                "client_host": str(getattr(client, "host", "") or ""),
                "skill": str(payload.get("skill") or ""),
                "granted_key_count": int(payload.get("granted_key_count") or 0),
                "granted_permission_count": int(payload.get("granted_permission_count") or 0),
                "extension_action": str(payload.get("extension_action") or ""),
                "extension_reason": str(payload.get("extension_reason") or ""),
            },
        )
    except Exception:
        log.debug("Failed to write owner grant audit event", exc_info=True)


def _grant_items_from_body(body: Dict[str, Any]) -> list[str]:
    raw = body.get("items")
    if raw is None:
        raw = body.get("keys")
    if raw is None:
        raw = body.get("granted_keys")
    if raw is None:
        return []
    out: list[str] = []
    values = raw if isinstance(raw, list) else [raw]
    for item in values:
        if isinstance(item, dict):
            value = item.get("value") or item.get("key") or item.get("permission") or item.get("name")
        else:
            value = item
        text = str(value or "").strip()
        if text and text not in out:
            out.append(text)
    return out


async def api_extensions_index(request: Request) -> JSONResponse:
    """Return discovered extensions plus live loader snapshot.

    The synchronous body runs in a worker thread and reuses discovered skills
    to avoid repeated filesystem walks during Widgets/Skills refresh.
    """
    try:
        import asyncio

        from ouroboros.config import get_skills_repo_path
        from ouroboros.skill_review_runner import reconcile_stale_review_jobs

        drive_root = _request_drive_root(request)
        repo_path = get_skills_repo_path()
        await asyncio.to_thread(
            reconcile_stale_review_jobs,
            drive_root,
            repo_path=repo_path,
        )
        payload = await asyncio.to_thread(_build_extensions_index, drive_root, repo_path)
        return JSONResponse(payload)
    except Exception as exc:
        log.exception("api_extensions_index failure")
        return json_exception(exc)


async def api_skill_daemons(_request: Request) -> JSONResponse:
    """Return host-supervised extension companion process status."""
    try:
        from ouroboros.extension_companion import snapshot_processes

        return JSONResponse({"companions": snapshot_processes()})
    except Exception as exc:
        return json_exception(exc)


def _build_extensions_index(drive_root, repo_path):
    """Threaded, request-scope-free body for ``GET /api/extensions``."""
    from ouroboros.extension_loader import extension_name_prefix, runtime_state_for_loaded_skill

    live_snapshot = snapshot()
    # Scan data plane plus optional external checkout; bootstrap copies native refs.
    skills = discover_skills(drive_root, repo_path=repo_path)
    unique_skills = [
        skill for skill in skills
        if not bool(getattr(skill, "identity_collision", False))
    ]
    try:
        from supervisor.queue import sync_skill_schedules

        # Empty inventory retires vanished skills; collision placeholders keep
        # prior rows for ambiguous identities while unique peers still sync.
        sync_skill_schedules(skills, drive_root=drive_root)
    except Exception:
        log.debug("Failed to sync skill schedules", exc_info=True)
    runtime_states = {
        s.name: runtime_state_for_loaded_skill(s, drive_root, skills=skills)
        for s in unique_skills
        if s.manifest.is_extension()
    }

    def _live_tool_count(skill_name: str) -> int:
        prefix = extension_name_prefix(skill_name)
        return sum(1 for name in live_snapshot.get("tools", []) if str(name).startswith(prefix))

    def _live_route_count(skill_name: str) -> int:
        prefix = f"/api/extensions/{skill_name}/"
        return sum(1 for name in live_snapshot.get("routes", []) if str(name).startswith(prefix))

    def _live_ws_count(skill_name: str) -> int:
        prefix = extension_name_prefix(skill_name)
        return sum(1 for name in live_snapshot.get("ws_handlers", []) if str(name).startswith(prefix))

    # Inline ClawHub provenance so Installed UI avoids a second round-trip.
    try:
        from ouroboros.marketplace.provenance import read_provenance, read_publication_record
    except Exception:  # pragma: no cover — defensive
        read_provenance = lambda *_a, **_kw: None  # type: ignore[assignment]
        read_publication_record = lambda *_a, **_kw: (None, None)  # type: ignore[assignment]
    marketplace_enabled = True

    catalog = []

    def _path_installed_at(skill_dir: pathlib.Path) -> str:
        candidates = [skill_dir / "SKILL.md", skill_dir / "plugin.py", skill_dir]
        stamps: list[float] = []
        for candidate in candidates:
            try:
                if candidate.exists():
                    stamps.append(candidate.stat().st_mtime)
            except OSError:
                continue
        if not stamps:
            return ""
        return datetime.fromtimestamp(min(stamps), tz=timezone.utc).isoformat().replace("+00:00", "Z")

    from ouroboros.extension_health import read_extension_health
    from ouroboros.gateway.presence_settings import presence_runtime_card_projection
    from ouroboros.skill_review_runner import skill_review_ui_projection
    from ouroboros.tools.github import github_token_from_env_or_settings

    # Request-invariant: resolve the github-token state ONCE for the whole index, not
    # once per skill (FR1 — avoids N settings.json reads per GET /api/extensions).
    _gh_token_configured = (
        bool(github_token_from_env_or_settings())
        if unique_skills
        else False
    )

    for s in skills:
        payload_root = ""
        try:
            rel_skill_dir = s.skill_dir.resolve().relative_to(drive_root.resolve())
            if rel_skill_dir.parts[:1] == ("skills",):
                payload_root = rel_skill_dir.as_posix()
        except Exception:
            payload_root = ""
        entry: dict[str, Any] = {
            "name": s.name,
            "type": s.manifest.type,
            "version": s.manifest.version,
            "description": s.manifest.description,
            "enabled": s.enabled,
            "permissions": list(s.manifest.permissions or []),
            "conflicts": list(getattr(s.manifest, "conflicts", []) or []),
            "load_error": s.load_error,
            "is_self_authored": bool(getattr(s, "is_self_authored", False)),
            # Keep source explicit so marketplace skills are not mislabeled native.
            "source": s.source,
            # Loader payload hash (§7.2): the hub UI's CAS/sync fact. Empty for
            # broken/collision rows, whose loader hash never existed.
            "content_hash": str(getattr(s, "content_hash", "") or ""),
            "payload_root": payload_root,
            "installed_at": _path_installed_at(s.skill_dir),
        }
        if bool(getattr(s, "identity_collision", False)):
            stale = True
            gate = skill_review_gate(s.review.status, stale=stale, findings=s.review.findings)
            # Serialize the collision fact itself: hub_sync must fail closed
            # (no-action conflict card) instead of first-wins joining one of
            # several same-name occupants (scope-review reproduction).
            entry["identity_collision"] = True
            entry.update({
                "review_status": s.review.status,
                "review_stale": stale,
                "review_gate": gate,
                "executable_review": False,
                "review_profile": "",
                "official_hub_verified": False,
                "owner_attestable": False,
                "submit_hub": _passive_submit_hub(
                    s,
                    github_token_configured=False,
                ),
                "conflict": None,
                "desired_live": False,
                "live_loaded": False,
                "live_reason": "load_error",
                "health_regressed": False,
                "last_known_good": None,
                "dispatch_live": False,
                "review_findings": [],
                "skill_review": {},
                "grants": {},
            })
            catalog.append(entry)
            continue

        health = read_extension_health(drive_root, s.name) if s.manifest.is_extension() else None
        entry.update({
            **_review_fields(s, github_token_configured=_gh_token_configured),
            "conflict": skill_conflict_status(s, skills),
            "load_error": runtime_states.get(s.name, {}).get("load_error", s.load_error),
            "desired_live": runtime_states.get(s.name, {}).get("desired_live", False),
            "live_loaded": runtime_states.get(s.name, {}).get("live_loaded", False),
            "live_reason": runtime_states.get(s.name, {}).get("reason", "not_extension"),
            "health_regressed": bool((health or {}).get("regressed")),
            "last_known_good": (health or {}).get("last_known_good"),
            "dispatch_live": bool(
                _live_tool_count(s.name)
                or _live_route_count(s.name)
                or _live_ws_count(s.name)
            ),
            "review_findings": list(s.review.findings or []),
            "skill_review": skill_review_ui_projection(drive_root, s.name),
            "grants": grant_status_for_skill(drive_root, s),
        })
        presence_runtime = presence_runtime_card_projection(drive_root, s)
        if presence_runtime is not None:
            entry["presence_runtime"] = presence_runtime
        # Durable OuroborosHub publication receipt (state-plane, survives bucket
        # moves). Collision rows above deliberately never read state, so these
        # fields are absent there. published=null when no valid record exists;
        # published_malformed=true when the file exists but fails validation.
        try:
            published, published_diagnostic = read_publication_record(drive_root, s.name)
        except Exception:  # pragma: no cover — defensive
            published, published_diagnostic = None, None
        entry["published"] = published
        entry["published_malformed"] = published_diagnostic is not None
        if s.source == "clawhub":
            try:
                prov = read_provenance(drive_root, s.name) or {}
            except Exception:  # pragma: no cover
                prov = {}
            if prov:
                if prov.get("installed_at"):
                    entry["installed_at"] = str(prov.get("installed_at") or "")
                entry["provenance"] = {
                    "slug": prov.get("slug", ""),
                    "version": prov.get("version", ""),
                    "sha256": prov.get("sha256", ""),
                    "adapter_version": prov.get("adapter_version", ""),
                    "openclaw_compat": dict(prov.get("openclaw_compat") or {}),
                    "installed_at": prov.get("installed_at", ""),
                    "updated_at": prov.get("updated_at", ""),
                }
                if marketplace_enabled:
                    entry["provenance"].update({
                        "homepage": prov.get("homepage", ""),
                        "license": prov.get("license", ""),
                        "primary_env": prov.get("primary_env", ""),
                        "adapter_warnings": list(prov.get("adapter_warnings") or []),
                        "original_manifest_sha256": prov.get("original_manifest_sha256", ""),
                        "translated_manifest_sha256": prov.get("translated_manifest_sha256", ""),
                        "registry_url": prov.get("registry_url", ""),
                    })
        catalog.append(entry)
    return {"skills": catalog, "live": live_snapshot}


async def api_extension_manifest(request: Request) -> JSONResponse:
    """GET /api/extensions/<skill>/manifest — raw manifest metadata."""
    from ouroboros.config import get_skills_repo_path
    from ouroboros.extension_loader import runtime_state_for_skill_name

    skill_name = str(request.path_params.get("skill") or "").strip()
    if not skill_name:
        return json_error("missing skill name", 400)
    drive_root = _request_drive_root(request)
    repo_path = get_skills_repo_path()
    loaded = await asyncio.to_thread(find_skill, drive_root, skill_name, repo_path=repo_path)
    if loaded is None:
        return json_error("skill not found", 404)
    runtime_state = await asyncio.to_thread(
        runtime_state_for_skill_name,
        skill_name,
        drive_root,
        repo_path=repo_path,
    )
    load_error = runtime_state.get("load_error")
    if not isinstance(load_error, str) or not load_error.strip():
        load_error = loaded.load_error
    return JSONResponse(
        {
            "name": loaded.name,
            "manifest": {
                "name": loaded.manifest.name,
                "description": loaded.manifest.description,
                "version": loaded.manifest.version,
                "type": loaded.manifest.type,
                "entry": loaded.manifest.entry,
                "permissions": list(loaded.manifest.permissions or []),
                "conflicts": list(getattr(loaded.manifest, "conflicts", []) or []),
                "env_from_settings": list(loaded.manifest.env_from_settings or []),
                "scheduled_tasks": list(getattr(loaded.manifest, "scheduled_tasks", []) or []),
                "ui_tab": loaded.manifest.ui_tab,
            },
            "enabled": loaded.enabled,
            **_review_fields(loaded),
            "content_hash": loaded.content_hash,
            "load_error": load_error,
        }
    )


async def api_extension_module(request: Request) -> Response:
    """Serve one reviewed JavaScript file of a live module widget from the loaded bundle.

    ``{entry:path}`` is a POSIX path relative to the skill directory: the
    declared entry or any sibling ``.js``/``.mjs`` the reviewed payload ships
    (``lib/x.js``). Authorization and content are one loader read under one
    lock: 409 when the skill has no live bundle; 404 when the path is not among
    the files captured when its module tab registered (dependency, cache, and
    dot-prefixed paths are never captured); 400 for a path with a backslash or
    NUL, an empty/``.``/``..`` segment (the ASGI server already decoded
    ``%2e%2e`` and ``%2F``), or a non-``.js``/``.mjs`` suffix. The body is the
    text captured at load — no per-request disk read, so an edit after load is
    not served until the skill reloads (DEVELOPMENT "Passive GET"). The
    requesting ``srcdoc`` frame has an opaque origin and fetches anonymously
    cross-origin, hence ``Access-Control-Allow-Origin: *`` (no credentials) on
    every response, refusals included — else ``import()`` sees a CORS failure.
    """
    from ouroboros.extension_loader import live_module_sources

    headers = {"Cache-Control": "no-store", "Access-Control-Allow-Origin": "*"}

    def refuse(message: str, status: int) -> Response:
        return JSONResponse({"error": message}, status_code=status, headers=headers)

    skill_name = str(request.path_params.get("skill") or "").strip()
    path = str(request.path_params.get("entry") or "")
    if (
        not skill_name or "\\" in path or "\0" in path
        or any(part in {"", ".", ".."} for part in path.split("/"))
        or not path.endswith((".js", ".mjs"))
    ):
        return refuse("invalid module path", 400)
    sources = live_module_sources(skill_name)
    if sources is None:
        return refuse(f"extension {skill_name!r} not live", 409)
    source = sources.get(path)
    if source is None:
        return refuse("module path is not a reviewed JavaScript file of a live widget", 404)
    return Response(source, media_type="application/javascript; charset=utf-8", headers=headers)


async def api_extension_settings_section(request: Request) -> JSONResponse:
    """Return declarative Settings sections registered by one extension."""
    skill_name = str(request.path_params.get("skill") or "").strip()
    if not skill_name:
        return json_error("missing skill name", 400)
    live = snapshot()
    sections = [
        item
        for item in live.get("settings_sections", [])
        if str(item.get("skill") or "") == skill_name
    ]
    return JSONResponse({"skill": skill_name, "sections": sections})


async def api_extension_dispatch(request: Request) -> Response:
    """Dispatch an extension route after reconciling live loader state."""
    from ouroboros.config import get_skills_repo_path, load_settings
    from ouroboros.extension_loader import reconcile_extension, runtime_state_for_skill_name

    skill = str(request.path_params.get("skill") or "").strip()
    rest = str(request.path_params.get("rest") or "").strip()
    mount = f"/api/extensions/{skill}/{rest}"
    drive_root = _request_drive_root(request)
    repo_path = get_skills_repo_path()
    spec = list_routes().get(mount)
    if spec is None and skill:
        state = await asyncio.to_thread(
            runtime_state_for_skill_name,
            skill,
            drive_root,
            repo_path=repo_path,
        )
        if state.get("desired_live"):
            state = await asyncio.to_thread(
                reconcile_extension,
                skill,
                drive_root,
                load_settings,
                repo_path=repo_path,
            )
            spec = list_routes().get(mount)
            if spec is None and state.get("action") == "extension_load_error":
                return json_error(f"extension {skill!r} failed to go live", 409, state=state)
        elif state.get("reason") != "missing":
            return json_error(f"extension {skill!r} not live: {state.get('reason')}", 409, state=state)
    if spec is None:
        return json_error(f"no extension route registered for {mount!r}", 404)
    state = await asyncio.to_thread(
        runtime_state_for_skill_name,
        str(spec.get("skill") or skill),
        drive_root,
        repo_path=repo_path,
    )
    if not state.get("desired_live") or not state.get("live_loaded"):
        state = await asyncio.to_thread(
            reconcile_extension,
            skill,
            drive_root,
            load_settings,
            repo_path=repo_path,
        )
        spec = list_routes().get(mount)
        if state.get("action") == "extension_load_error":
            return json_error(f"extension {skill!r} failed to go live", 409, state=state)
    if not state.get("desired_live") or not state.get("live_loaded"):
        return json_error(f"extension {skill!r} not live: {state.get('reason')}", 409, state=state)
    if spec is None:
        return json_error(f"no extension route registered for {mount!r}", 404)
    method = request.method.upper()
    allowed = {m.upper() for m in spec.get("methods", ("GET",))}
    if "GET" in allowed:
        allowed.add("HEAD")
    if method not in allowed:
        return json_error(f"method {method} not allowed; allowed={sorted(allowed)}", 405)
    if spec.get("out_of_process"):
        try:
            from ouroboros.extension_process_runner import dispatch_extension_route_subprocess

            try:
                body = await _read_child_dispatch_body(request)
            except ValueError as exc:
                return json_error(str(exc), 413)
            headers = [
                (key, value)
                for key, value in request.headers.items()
                if key.lower() not in _CHILD_DISPATCH_HEADER_DENYLIST
            ]
            child_result = await asyncio.to_thread(
                dispatch_extension_route_subprocess,
                spec,
                {
                    "method": method,
                    "path": request.url.path,
                    "path_params": dict(request.path_params),
                    "query_string": request.url.query,
                    "headers": headers,
                    "body_b64": base64.b64encode(body).decode("ascii"),
                },
                drive_root=drive_root,
                repo_dir=_request_repo_dir(request),
            )
            route_result = dict(child_result.get("route") or {})
            kind = str(route_result.get("kind") or "")
            status_code = int(route_result.get("status_code") or 200)
            if kind == "response":
                headers = dict(route_result.get("headers") or {})
                headers.pop("content-length", None)
                body_bytes = base64.b64decode(str(route_result.get("body_b64") or ""))
                return Response(
                    body_bytes,
                    status_code=status_code,
                    headers=headers,
                    media_type=route_result.get("media_type") or None,
                )
            if kind == "json":
                return JSONResponse(route_result.get("data"), status_code=status_code)
            return Response(str(route_result.get("text") or ""), status_code=status_code)
        except Exception as exc:
            log.exception("extension child dispatch failure: %s", mount)
            return json_error(f"{type(exc).__name__}: {exc}", 502)
    handler = spec.get("handler")
    if not callable(handler):
        return json_error("registered handler is not callable")
    try:
        from ouroboros.extension_process_runner import disclose_inprocess_extension_dispatch

        disclose_inprocess_extension_dispatch(
            spec,
            drive_root=drive_root,
            surface_kind="route",
            surface=mount,
        )
    except Exception as exc:
        log.exception("extension cost disclosure failure: %s", mount)
        return json_error(f"model-cost disclosure failed: {type(exc).__name__}: {exc}", 502)
    try:
        if inspect.iscoroutinefunction(handler):
            result = await handler(request)
        else:
            result = await asyncio.to_thread(handler, request)
        if inspect.iscoroutine(result):
            result = await result
    except Exception as exc:
        log.exception("extension dispatch failure: %s", mount)
        return json_error(f"{type(exc).__name__}: {exc}")
    if isinstance(result, Response):
        return result
    return JSONResponse(result if result is not None else {})


async def api_skill_toggle(request: Request) -> JSONResponse:
    """Toggle a skill from the UI and run extension load/unload reconciliation."""
    from ouroboros.config import get_skills_repo_path, load_settings
    from ouroboros.skill_loader import find_skill, grant_status_for_skill, save_enabled
    from ouroboros import extension_loader

    skill_name = str(request.path_params.get("skill") or "").strip()
    if not skill_name:
        return json_error("missing skill name", 400)
    body = await request_json_or(request, {}, exceptions=(Exception,))
    bool_sentinel = object()
    enabled = coerce_bool(body.get("enabled"), default=bool_sentinel)
    if enabled is bool_sentinel:
        return json_error("'enabled' must be a boolean", 400)

    drive_root = _request_drive_root(request)
    repo_path = get_skills_repo_path()

    initial = await asyncio.to_thread(find_skill, drive_root, skill_name, repo_path=repo_path)
    if initial is None:
        return json_error("skill not found", 404)
    def _run_toggle_sync() -> dict[str, Any]:
        loaded = find_skill(drive_root, skill_name, repo_path=repo_path)
        if loaded is None:
            return {"error": "skill not found", "status_code": 404}
        collision_load_error = loaded.load_error.lower().startswith("skill name collision:")
        if enabled and loaded.load_error:
            return {"error": f"cannot enable: {loaded.load_error}", "status_code": 400}
        if enabled:
            conflict = skill_conflict_status(
                loaded,
                discover_skills(drive_root, repo_path=repo_path),
            )
            if conflict:
                names = list(conflict.get("skills") or [])
                return {
                    "error": (
                        "cannot enable while conflicting skills are enabled: "
                        + ", ".join(names)
                    ),
                    "status_code": 409,
                    "conflict": conflict,
                }
            stale = loaded.review.is_stale_for(loaded.content_hash)
            grants = grant_status_for_skill(drive_root, loaded)
            gate = skill_review_gate(loaded.review.status, stale=stale, findings=loaded.review.findings)
            if not gate["executable_review"]:
                return {
                    "error": "cannot enable until review status is a fresh executable review",
                    "status_code": 409,
                    **_review_fields(loaded, stale=stale, gate=gate),
                    "grants": grants,
                }
            if not grants.get("all_granted", True):
                return {
                    "error": "cannot enable until requested key and permission grants are approved",
                    "status_code": 409,
                    **_review_fields(loaded, stale=stale, gate=gate),
                    "grants": grants,
                }
            # Mirror toggle_skill's isolated-dependency enable guard for the UI.
            try:
                from ouroboros.marketplace.install_specs import install_specs_hash
                from ouroboros.marketplace.isolated_deps import read_deps_state
                from ouroboros.skill_dependencies import auto_install_specs_for_skill

                auto_specs = auto_install_specs_for_skill(drive_root, loaded)
                if auto_specs:
                    deps_state = read_deps_state(drive_root, loaded.name, loaded.skill_dir)
                    deps_status = str(deps_state.get("status") or "pending")
                    expected_hash = install_specs_hash(auto_specs)
                    actual_hash = str(deps_state.get("specs_hash") or "")
                    if deps_status != "installed":
                        return {
                            "error": "cannot enable until isolated dependencies are installed",
                            "status_code": 409,
                            "deps_status": deps_status,
                            "deps_error": deps_state.get("error", ""),
                            **_review_fields(loaded, stale=stale, gate=gate),
                            "grants": grants,
                        }
                    if actual_hash != expected_hash:
                        return {
                            "error": "cannot enable until isolated dependency fingerprint is refreshed",
                            "status_code": 409,
                            "deps_status": "stale",
                            **_review_fields(loaded, stale=stale, gate=gate),
                            "grants": grants,
                        }
            except Exception:
                log.debug("api_skill_toggle deps probe failed", exc_info=True)
        if not enabled and collision_load_error:
            action = None
            if loaded.name in extension_loader.snapshot()["extensions"]:
                extension_loader.unload_extension(loaded.name)
                action = "extension_unloaded"
            return {
                "error": (
                    "cannot persist disable because this skill's sanitized "
                    "name collides with another skill directory; rename one "
                    "of the directories first"
                ),
                "status_code": 400,
                "extension_action": action,
                "extension_reason": "name_collision",
            }
        save_enabled(drive_root, loaded.name, enabled)
        try:
            from supervisor.queue import sync_skill_schedules

            sync_skill_schedules(discover_skills(drive_root, repo_path=repo_path), drive_root=drive_root)
        except Exception:
            log.debug("api_skill_toggle schedule sync failed", exc_info=True)
        action = None
        live_reason = "not_extension"
        if loaded.manifest.is_extension() or loaded.name in extension_loader.snapshot()["extensions"]:
            state = extension_loader.reconcile_extension(
                loaded.name,
                drive_root,
                load_settings,
                repo_path=repo_path,
                retry_load_error=True,
                revert_enabled_on_error=enabled,
            )
            action = state.get("action")
            live_reason = str(state.get("reason") or "")
            if enabled and action == "extension_load_error":
                # Atomic enable: reconcile already reverted enabled.json after the real
                # out-of-process catalog/register dry-run failed, so the skill is never
                # left enabled-but-broken. Re-sync schedules to the reverted state and
                # surface the concrete load error.
                try:
                    from supervisor.queue import sync_skill_schedules
                    sync_skill_schedules(discover_skills(drive_root, repo_path=repo_path), drive_root=drive_root)
                except Exception:
                    log.debug("api_skill_toggle revert schedule sync failed", exc_info=True)
                return {
                    "error": f"cannot enable: {state.get('load_error') or 'extension failed to load'}",
                    "status_code": 409,
                    "skill": loaded.name,
                    "source": loaded.source,
                    **_review_fields(loaded),
                    "grants": grant_status_for_skill(drive_root, loaded),
                    "extension_action": action,
                    "extension_reason": live_reason,
                }
        return {
            "skill": loaded.name,
            "source": loaded.source,
            **_review_fields(loaded),
            "grants": grant_status_for_skill(drive_root, loaded),
            "action": action,
            "live_reason": live_reason,
        }

    async def _run_toggle() -> dict[str, Any]:
        return await run_blocking_preserving_cancellation(
            _run_toggle_sync,
            log_label="skill toggle lifecycle operation",
        )

    queued = await run_lifecycle_job(
        kind="enable" if enabled else "disable",
        target=initial.name,
        source=initial.source,
        message=("Enabling" if enabled else "Disabling") + f" {initial.name}",
        runner=_run_toggle,
        options=LifecycleJobOptions(
            drive_root=drive_root,
            result_message=lambda item: (
                item.get("error", "")
                or (("Enabled" if enabled else "Disabled") + f" {item.get('skill', initial.name)}")
            ),
            result_error=lambda item: item.get("error", ""),
        ),
    )
    if queued.get("error"):
        return JSONResponse(queued, status_code=int(queued.get("status_code") or 400))
    _broadcast_extension_lifecycle(
        request,
        str(queued.get("skill") or initial.name),
        queued.get("action"),
        queued.get("live_reason"),
    )
    return JSONResponse(
        {
            "skill": queued.get("skill", initial.name),
            "enabled": enabled,
            "review_status": queued.get("review_status"),
            "review_stale": queued.get("review_stale"),
            "review_gate": queued.get("review_gate"),
            "executable_review": queued.get("executable_review"),
            "grants": queued.get("grants", {}),
            "extension_action": queued.get("action"),
            "extension_reason": queued.get("live_reason"),
        }
    )


class _ApiReviewCtx:
    """Minimal ToolContext-compatible carrier for HTTP-triggered review."""

    def __init__(self, drive_root: pathlib.Path, repo_dir: pathlib.Path) -> None:
        self.drive_root = drive_root
        self.repo_dir = repo_dir
        self.task_id = "api_skill_review"
        self.current_chat_id = 0
        self.pending_events: list = []
        self.emit_progress_fn = None
        self.event_queue = None  # _emit_usage_event falls back to pending_events
        self.messages: list = []


async def api_skill_review(request: Request) -> JSONResponse:
    """Queue tri-model skill review from the UI without blocking the event loop."""
    skill_name = str(request.path_params.get("skill") or "").strip()
    if not skill_name:
        return json_error("missing skill name", 400)

    drive_root = _request_drive_root(request)
    repo_dir = _request_repo_dir(request)
    ctx = _ApiReviewCtx(drive_root, repo_dir)
    from ouroboros.skill_review_runner import run_skill_review_lifecycle
    from ouroboros.skill_review import review_skill as _review_skill_impl

    payload = await run_skill_review_lifecycle(
        ctx,
        skill_name,
        source="skills",
        review_impl=_review_skill_impl,
    )
    return JSONResponse(payload)


async def api_owner_skill_attest_review(request: Request) -> JSONResponse:
    """POST /api/owner/skills/{skill}/attest-review — OWNER-ONLY (C1, v6.39): skip the
    EXPENSIVE LLM review for the owner's own external/self-authored skill or for a freshly
    hash-verified official OuroborosHub payload. The DETERMINISTIC preflight floor still runs
    (409 if it fails); only the costly LLM phase is skipped. Loudly audited. The agent can
    never reach this — the owner_attestation marker is an agent-write-protected owner-state
    file, so this is owner-issued only."""
    skill_name = str(request.path_params.get("skill") or "").strip()
    if not skill_name:
        return json_error("missing skill name", 400)
    drive_root = _request_drive_root(request)
    repo_dir = _request_repo_dir(request)
    ctx = _ApiReviewCtx(drive_root, repo_dir)
    from ouroboros.skill_review_runner import run_skill_review_lifecycle
    from ouroboros.skill_owner_attestation import review_skill_owner_attest

    # Route through the SAME lifecycle as api_skill_review so a clean attestation gets the
    # post-pass deps/extension reconcile + schedule resync (otherwise an attested skill with
    # isolated dependencies stays blocked by skill_readiness). The lifecycle just calls our
    # attest impl instead of the LLM review.
    payload = await run_skill_review_lifecycle(
        ctx, skill_name, source="skills", review_impl=review_skill_owner_attest,
    )
    status = str(payload.get("status") or "")
    try:
        append_jsonl(pathlib.Path(drive_root) / "logs" / "events.jsonl", {
            "ts": utc_now_iso(),
            "type": "owner_api_action",
            "action": "skill_owner_attest",
            "client_host": str(getattr(getattr(request, "client", None), "host", "") or ""),
            "skill": skill_name,
            "status": status,
            "content_hash": str(payload.get("content_hash") or ""),
        })
    except Exception:
        log.debug("Failed to write owner attestation audit event", exc_info=True)
    if status != "clean":
        # Deterministic preflight floor failed, the skill is not owner-own, or it could not
        # be loaded/hashed: 409 — not attestable. A failed preflight persists as the recorded
        # review result when review.json was absent/stale; a fresh valid verdict stays untouched.
        return JSONResponse(payload, status_code=409)
    return JSONResponse(payload)


async def api_skill_lifecycle_queue(request: Request) -> JSONResponse:
    """GET /api/skills/lifecycle-queue — recent mutating skill operations."""

    try:
        from ouroboros.skill_review_runner import reconcile_stale_review_jobs

        await asyncio.to_thread(reconcile_stale_review_jobs, _request_drive_root(request))
    except Exception:
        log.debug("stale review job reconciliation failed", exc_info=True)
    return JSONResponse(queue_snapshot())


def _skill_review_history_detail_sync(
    drive_root: pathlib.Path, skill_name: str, job_id: str,
) -> Dict[str, Any]:
    """Locate ONE terminal review record by job_id and render its markdown.

    Read-only over the append-only ``review_history.jsonl``. Raw reviewer text
    never leaves the history file: findings are the already-normalized
    ``parsed_items`` rows, and degraded (non-responsive) reviewers are
    disclosed by model + status with a pointer instead of the raw body.
    """
    from ouroboros import skill_review_history
    from ouroboros.skill_review import render_skill_review_block
    from ouroboros.skill_review_status import (
        STATUS_BLOCKERS,
        STATUS_CLEAN,
        STATUS_PENDING,
        STATUS_WARNINGS,
    )

    record, lookup_status = skill_review_history.find_history_job_bounded(
        drive_root, skill_name, job_id,
    )
    if lookup_status == "absent":
        return {"error": "no review history for skill", "status_code": 404}
    if lookup_status == "io_error":
        return {
            "error": "review history is temporarily unavailable; retry the detail",
            "status_code": 503,
        }
    if record is None:
        error = (
            "review record unavailable outside the bounded history window"
            if lookup_status == "unavailable"
            else "review record not found"
        )
        return {"error": error, "status_code": 404}
    raw_actors = [
        actor for actor in (record.get("raw_actor_records") or [])
        if isinstance(actor, dict)
    ]
    actor_models = [
        str(actor.get("model_id") or actor.get("model") or "reviewer")
        for actor in raw_actors
    ]
    duplicate_models = {
        model for model in actor_models if actor_models.count(model) > 1
    }
    duplicate_occurrences: Dict[str, int] = {}
    labeled_actors = []
    for actor in raw_actors:
        model = str(actor.get("model_id") or actor.get("model") or "reviewer")
        slot_id = str(actor.get("slot_id") or "")
        duplicate_occurrences[model] = duplicate_occurrences.get(model, 0) + 1
        qualifier = slot_id or f"legacy-actor-{duplicate_occurrences[model]}"
        label = f"{model} [{qualifier}]" if model in duplicate_models else model
        labeled_actors.append((actor, label))
    findings = [
        {**item, "model": label}
        for actor, label in labeled_actors
        for item in (actor.get("parsed_items") or [])
        if isinstance(item, dict)
    ]
    reviewer_models = [
        label for _actor, label in labeled_actors
    ]
    degraded_actors = [
        {
            "model_id": label,
            "status": str(actor.get("status") or "unknown"),
            "raw_text": "(raw reviewer output withheld from chat; stored in review_history.jsonl)",
        }
        for actor, label in labeled_actors
        if str(actor.get("status") or "") != "responded"
    ]
    status = str(record.get("status") or "pending")
    terminal_reason = str(record.get("terminal_reason") or "")
    lifecycle_status = str(
        record.get("job_status") or record.get("lifecycle_status") or ""
    ).strip().lower()
    lifecycle_failed = lifecycle_status in {
        "failed", "error", "timeout", "interrupted", "cancelled",
    }
    # Interrupted/timeout/failed records carry no review verdict; surface the
    # terminal reason honestly instead of pretending a review body exists.
    error_note = (
        terminal_reason or lifecycle_status or status
        if lifecycle_failed
        else ("" if status in {
            STATUS_CLEAN, STATUS_WARNINGS, STATUS_BLOCKERS, STATUS_PENDING,
        } else (terminal_reason or status))
    )
    attempt = int(record.get("snapshot_attempt") or 1)
    outcome = {
        "skill": skill_name,
        "status": status,
        "content_hash": str(record.get("content_hash") or ""),
        "findings": findings,
        "reviewer_models": reviewer_models,
        "review_round": int(record.get("review_round") or attempt),
        "snapshot_attempt": attempt,
        "snapshot_revised": bool(record.get("snapshot_revised")),
        "raw_actor_records": degraded_actors,
        "error": error_note,
    }
    markdown = render_skill_review_block(outcome, attempt_idx=attempt)
    if degraded_actors:
        markdown += (
            "\n\n_A terminal reviewer slot that never started or refused has no "
            "physical-attempt ledger row; incomplete attempt coverage can therefore be final._"
        )
    # Max-Review-Cycles accounting facts (Q16/Q17 auditability) ride the
    # free-form markdown detail: the response contract
    # (SkillReviewHistoryDetailResponse) is typed to exactly four fields, so
    # adding response keys would need an api_types version bump — the rendered
    # detail string is the additive channel. Legacy rows without the facts
    # render nothing.
    accounting = []
    usage_detail = ""
    replayed = bool(record.get("replayed_from_ts"))
    if replayed:
        accounting.append(
            f"free replay of the {record.get('replayed_from_ts')} verdict; "
            "no physical reviewer dispatch for this replay"
        )
    elif record.get("paid"):
        accounting.append("paid panel dispatch (counts toward Max Review Cycles)")
        if record.get("usage_attribution_schema") == "physical_attempt_v1":
            from ouroboros.usage_accounting import skill_review_usage

            try:
                usage = skill_review_usage(
                    drive_root, review_skill=skill_name,
                    review_wave_id=str(record.get("wave_id") or job_id),
                )
                if usage.get("attempt_ids"):
                    known, expected, recorded = skill_review_attempt_coverage(record, usage)
                    usage_detail = skill_review_usage_markdown(
                        usage, coverage_known=known, expected=expected, recorded=recorded,
                    )
                else:
                    accounting.append(
                        "no canonical physical-attempt rows are recorded yet; "
                        "cash and finality are unavailable"
                    )
            except Exception:
                log.debug("skill review physical-attempt detail unavailable", exc_info=True)
                accounting.append("exact physical-attempt accounting is currently unavailable")
        else:
            accounting.append(
                "exact per-wave physical-attempt attribution was unavailable in this version"
            )
    if record.get("review_contract_fingerprint"):
        accounting.append(
            f"panel contract {str(record.get('review_contract_fingerprint'))[:12]}…"
        )
    if record.get("rebuttal_sha256"):
        accounting.append(f"rebuttal sha256 {str(record.get('rebuttal_sha256'))[:12]}…")
    if accounting:
        markdown += "\n\n_Review accounting: " + "; ".join(accounting) + "._"
    if usage_detail:
        markdown += "\n\n" + usage_detail
    elif replayed:
        markdown += "\n\n_Cost: $0 (free replay)._"
    else:
        markdown += "\n\n_Cost unavailable._"
    return {
        "markdown": markdown,
        "status": status,
        "content_hash": outcome["content_hash"],
        "job_status": str(record.get("job_status") or ""),
    }


async def api_skill_review_history_detail(request: Request) -> JSONResponse:
    """GET /api/skills/{skill}/review-history/{job_id} — lazy Chat-card detail.

    Serves the server-rendered normalized review block for the exact terminal
    record a ``skill_review`` chat row references, so the compact reference
    row can expand without republishing review bodies into ``chat.jsonl``.
    """
    skill_name = str(request.path_params.get("skill") or "").strip()
    job_id = str(request.path_params.get("job_id") or "").strip()
    if not skill_name or not job_id:
        return json_error("missing skill or job id", 400)
    if _sanitize_skill_name(skill_name) != skill_name:
        return json_error("unknown skill", 404)
    try:
        payload = await asyncio.to_thread(
            _skill_review_history_detail_sync,
            _request_drive_root(request), skill_name, job_id,
        )
    except Exception as exc:
        return json_exception(exc)
    if payload.get("error"):
        return json_error(
            str(payload["error"]), int(payload.get("status_code") or 500),
        )
    return JSONResponse(payload)


async def api_skill_grants(request: Request) -> JSONResponse:
    """Owner grant path for reviewed skill settings keys and host permissions."""
    from ouroboros import extension_loader
    from ouroboros.config import get_skills_repo_path, load_settings

    skill_name = str(request.path_params.get("skill") or "").strip()
    if not skill_name:
        return json_error("missing skill name", 400)
    body = await request_json_or(request, {}, exceptions=(Exception,))
    if not isinstance(body, dict):
        return json_error("request body must be a JSON object", 400)

    drive_root = _request_drive_root(request)
    repo_path = get_skills_repo_path()

    def _save_grants_sync() -> dict[str, Any]:
        loaded = find_skill(drive_root, skill_name, repo_path=repo_path)
        if loaded is None:
            return {"error": "skill not found", "status_code": 404}
        if not (loaded.manifest.is_script() or loaded.manifest.is_extension()):
            return {
                "error": "key and permission grants are supported for script and extension skills",
                "status_code": 400,
            }
        stale = loaded.review.is_stale_for(loaded.content_hash)
        gate = skill_review_gate(loaded.review.status, stale=stale, findings=loaded.review.findings)
        if not review_status_allows_execution(loaded.review.status) or stale:
            return {
                "error": "key and permission grants require a fresh executable review",
                "status_code": 409,
                **_review_fields(loaded, stale=stale, gate=gate),
                "grants": grant_status_for_skill(drive_root, loaded),
            }
        allowed_keys = requested_core_setting_keys(list(loaded.manifest.env_from_settings or []))
        allowed_permissions = requested_skill_permissions(
            list(getattr(loaded.manifest, "permissions", []) or []),
            list(getattr(loaded.manifest, "subscribe_events", []) or []),
        )
        permission_map = {permission.lower(): permission for permission in allowed_permissions}
        requested_raw = _grant_items_from_body(body)
        requested_keys: list[str] = []
        requested_permissions: list[str] = []
        rejected: list[str] = []
        for item in requested_raw:
            key = item.upper()
            permission = permission_map.get(item.lower())
            if key in allowed_keys:
                if key not in requested_keys:
                    requested_keys.append(key)
            elif permission:
                if permission not in requested_permissions:
                    requested_permissions.append(permission)
            else:
                rejected.append(item)
        if not requested_raw or rejected or (not requested_keys and not requested_permissions):
            return {
                "error": (
                    "grant items must be requested by the current manifest; "
                    f"allowed keys={allowed_keys}, permissions={allowed_permissions}"
                ),
                "status_code": 400,
                "allowed_keys": allowed_keys,
                "allowed_permissions": allowed_permissions,
                "rejected_items": rejected,
            }
        save_skill_grants(
            drive_root,
            loaded.name,
            requested_keys,
            content_hash=loaded.content_hash,
            requested_keys=allowed_keys,
            granted_permissions=requested_permissions,
            requested_permissions=allowed_permissions,
        )
        extension_action = None
        extension_reason = None
        extension_load_error = None
        if loaded.manifest.is_extension():
            try:
                state = extension_loader.reconcile_extension(
                    loaded.name,
                    drive_root,
                    load_settings,
                    repo_path=repo_path,
                    retry_load_error=True,
                )
                extension_action = state.get("action")
                extension_reason = state.get("reason")
                extension_load_error = state.get("load_error")
            except Exception as exc:
                log.warning(
                    "Skill grant saved but extension reconcile failed for %s: %s",
                    loaded.name,
                    exc,
                    exc_info=True,
                )
                extension_reason = "reconcile_call_failed"
                extension_load_error = str(exc)
        try:
            from supervisor.queue import sync_skill_schedules
            sync_skill_schedules(discover_skills(drive_root, repo_path=repo_path), drive_root=drive_root)
        except Exception:
            log.debug("api_skill_grants schedule sync failed", exc_info=True)
        refreshed = find_skill(drive_root, loaded.name, repo_path=repo_path) or loaded
        return {
            "ok": True,
            "skill": loaded.name,
            "granted_keys": requested_keys,
            "granted_permissions": requested_permissions,
            "extension_action": extension_action,
            "extension_reason": extension_reason,
            "load_error": extension_load_error,
            "grants": grant_status_for_skill(drive_root, refreshed),
        }

    result = await asyncio.to_thread(_save_grants_sync)
    if result.get("error"):
        return JSONResponse(result, status_code=int(result.get("status_code") or 400))
    _owner_grant_audit(
        drive_root,
        request,
        {
            "skill": result.get("skill"),
            "granted_key_count": len(result.get("granted_keys") or []),
            "granted_permission_count": len(result.get("granted_permissions") or []),
            "extension_action": result.get("extension_action"),
            "extension_reason": result.get("extension_reason"),
        },
    )
    _broadcast_extension_lifecycle(
        request,
        str(result.get("skill") or skill_name),
        result.get("extension_action"),
        result.get("extension_reason"),
    )
    return JSONResponse(result)


async def api_skill_reconcile(request: Request) -> JSONResponse:
    """Re-run the extension load gate after launcher-owned grants change."""
    from ouroboros.config import get_skills_repo_path, load_settings
    from ouroboros import extension_loader

    skill_name = str(request.path_params.get("skill") or "").strip()
    if not skill_name:
        return json_error("missing skill name", 400)

    drive_root = _request_drive_root(request)
    repo_path = get_skills_repo_path()
    state = await asyncio.to_thread(
        extension_loader.reconcile_extension,
        skill_name,
        drive_root,
        load_settings,
        repo_path=repo_path,
        retry_load_error=True,
    )
    _broadcast_extension_lifecycle(
        request,
        skill_name,
        state.get("action"),
        state.get("reason"),
    )
    # Reconcile can flip grants/load state, so refresh schedule readiness now.
    try:
        from supervisor.queue import resync_skill_schedules

        resync_skill_schedules(drive_root)
    except Exception:
        log.debug("api_skill_reconcile schedule sync failed", exc_info=True)
    return JSONResponse(
        {
            "skill": skill_name,
            "extension_action": state.get("action"),
            "extension_reason": state.get("reason"),
            "live_loaded": bool(state.get("live_loaded")),
            "load_error": state.get("load_error"),
        }
    )


async def api_skill_delete(request: Request) -> JSONResponse:
    """Delete a local data-plane skill payload and its durable state."""
    from ouroboros.config import get_skills_repo_path
    from ouroboros import extension_loader

    skill_name = _sanitize_skill_name(str(request.path_params.get("skill") or "").strip())
    if not skill_name or skill_name == "_unnamed":
        return json_error("missing skill name", 400)
    body = await request_json_or(request, {}, exceptions=(Exception,))

    drive_root = _request_drive_root(request)
    repo_path = get_skills_repo_path()

    def _run_delete_sync() -> dict[str, Any]:
        requested_root = str(body.get("payload_root") or f"skills/external/{skill_name}").strip()
        root_parts = pathlib.PurePosixPath(requested_root).parts
        if len(root_parts) != 3 or root_parts[:2] != ("skills", "external"):
            return {"error": "local skill delete requires payload_root=skills/external/<name>", "status_code": 403}

        drive_root_path = pathlib.Path(drive_root).absolute()
        skills_root = drive_root_path / "skills"
        external_root = skills_root / "external"
        payload_dir = external_root / root_parts[2]
        if skills_root.is_symlink() or external_root.is_symlink() or payload_dir.is_symlink():
            return {"error": "local skill delete refuses symlinked data/skills/external payloads", "status_code": 403}

        skills = discover_skills(drive_root, repo_path=repo_path)
        loaded = next((item for item in skills if pathlib.Path(item.skill_dir).absolute() == payload_dir), None)
        if loaded is None:
            return {"error": f"skill {skill_name!r} not found at {requested_root}", "status_code": 404}
        if loaded.name != skill_name or loaded.source not in {"self_authored", "external"}:
            return {"error": "local skill delete is limited to self-authored/external skills", "status_code": 403}
        if any(item.name == skill_name and pathlib.Path(item.skill_dir).absolute() != payload_dir for item in skills):
            return {
                "error": (
                    "refusing to delete a local skill while another skill uses the same sanitized name; "
                    "rename one of the colliding skills first"
                ),
                "status_code": 409,
            }

        state_root = (drive_root_path / "state" / "skills").absolute()
        state_dir = state_root / loaded.name
        if state_root.is_symlink() or state_dir.is_symlink():
            return {"error": f"refusing to delete unsafe state path for {loaded.name!r}", "status_code": 500}
        try:
            state_dir.relative_to(state_root)
        except ValueError:
            return {"error": f"refusing to delete unsafe state path for {loaded.name!r}", "status_code": 500}

        extension_loader.unload_extension(loaded.name)
        shutil.rmtree(payload_dir)
        deleted_state = state_dir.exists()
        if deleted_state:
            shutil.rmtree(state_dir)
        try:
            from supervisor.queue import sync_skill_schedules
            sync_skill_schedules(discover_skills(drive_root, repo_path=repo_path), drive_root=drive_root)
        except Exception:
            log.debug("api_skill_delete schedule sync failed", exc_info=True)
        if payload_dir.exists() or state_dir.exists():
            return {"error": f"failed to fully delete local skill {loaded.name!r}", "status_code": 500}
        return {
            "ok": True,
            "skill": loaded.name,
            "source": loaded.source,
            "deleted_payload_root": f"skills/external/{root_parts[2]}",
            "deleted_state": deleted_state,
            "extension_action": "extension_unloaded",
            "extension_reason": "deleted",
        }

    queued = await run_lifecycle_job(
        kind="delete",
        target=skill_name,
        source="external",
        message=f"Deleting {skill_name}",
        runner=lambda: run_blocking_preserving_cancellation(
            _run_delete_sync,
            log_label="local skill delete lifecycle operation",
        ),
        options=LifecycleJobOptions(
            drive_root=drive_root,
            result_message=lambda item: item.get("error", "") or f"Deleted {item.get('skill', skill_name)}",
            result_error=lambda item: item.get("error", ""),
        ),
    )
    if queued.get("error"):
        return JSONResponse(queued, status_code=int(queued.get("status_code") or 400))
    _broadcast_extension_lifecycle(
        request,
        str(queued.get("skill") or skill_name),
        queued.get("extension_action"),
        queued.get("extension_reason"),
    )
    return JSONResponse(queued)


__all__ = [
    "api_extensions_index",
    "api_extension_manifest",
    "api_extension_module",
    "api_extension_settings_section",
    "api_extension_dispatch",
    "api_skill_daemons",
    "api_skill_delete",
    "api_skill_toggle",
    "api_skill_review",
    "api_skill_grants",
    "api_skill_reconcile",
]

"""HTTP endpoints for extension catalogue, manifests, modules, and dispatch."""

from __future__ import annotations

import asyncio
import base64
import inspect
import logging
import pathlib
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
from ouroboros.gateway.extension_receipts import extension_process_receipt, extension_reconcile_receipt
from ouroboros.marketplace.ouroboroshub import display_catalog_files
from ouroboros.skill_lifecycle_queue import (
    queue_snapshot,
    run_blocking_preserving_cancellation,
)
from ouroboros.skill_loader import (
    discover_skills,
    find_skill,
    grant_status_for_skill,
    skill_conflict_status,
    skill_review_gate,
    _sanitize_skill_name,
)
from ouroboros.skill_review_usage import (
    skill_review_attempt_coverage,
    skill_review_usage_markdown,
)

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


def _passive_submit_hub(
    loaded: Any,
    *,
    github_token_configured: bool | None = None,
    review_stale: bool | None = None,
) -> dict[str, Any]:
    """Project passive visibility/admission without running the scanner."""
    from ouroboros.skill_publish_eligibility import (
        PUBLISHABLE_SOURCES,
        publication_author_acceptance,
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
        author_accepted=bool(publication_author_acceptance(loaded.review, loaded.content_hash)),
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
    hub_catalog_files: dict[str, Any] | None = None,
) -> dict[str, Any]:
    stale = loaded.review.is_stale_for(loaded.content_hash) if stale is None else stale
    gate = loaded.review.gate_for(loaded.content_hash) if gate is None else gate
    source = str(getattr(loaded, "source", "") or "")
    # Hub hint: matched against the display-plane catalog view (§7.1a) the caller
    # peeked, never fetched here — a local listing must not wait for the network.
    # None = no fresh view yet; the page re-reads once its own catalog read lands.
    # With a view, an unreadable payload is a definitive negative for THIS row —
    # never a 500 for the whole listing.
    official_hub_verified: bool | None = False
    if source == "ouroboroshub":
        from ouroboros.skill_review import hub_payload_matches

        if hub_catalog_files is None:
            official_hub_verified = None
        else:
            try:
                official_hub_verified = hub_payload_matches(loaded, hub_catalog_files.__getitem__)
            except Exception:
                official_hub_verified = False
    owner_attestable = official_hub_verified if source == "ouroboroshub" else (
        source not in {"native", "clawhub"}
        and (source == "external" or bool(getattr(loaded, "is_self_authored", False)))
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
        "author_disposition": dict(loaded.review.author_disposition),
        "reviewed_content_hash": gate["reviewed_content_hash"],
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
    to avoid repeated filesystem walks during Widgets/Skills refresh. A passive
    read (DEVELOPMENT "Passive GET"): a dead ``running`` review job is healed by
    its owners (boot, the maintenance pass, the next review start), never here.
    """
    try:
        from ouroboros.config import get_skills_repo_path

        drive_root = _request_drive_root(request)
        repo_path = get_skills_repo_path()
        payload = await asyncio.to_thread(_build_extensions_index, drive_root, repo_path)
        return JSONResponse(payload)
    except Exception as exc:
        return json_exception(exc, context="api_extensions_index failure")


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
    hub_catalog_files = display_catalog_files()  # one view per response, never a fetch
    # Scan data plane plus optional external checkout; bootstrap copies native refs.
    skills = discover_skills(drive_root, repo_path=repo_path)
    unique_skills = [
        skill for skill in skills
        if not bool(getattr(skill, "identity_collision", False))
    ]
    # No schedule sync here: it takes the supervisor queue lock and the schedule
    # file lock, so a read would wait on every holder. The scheduler tick and
    # the lifecycle actions mirror manifest schedules.
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
            **_review_fields(
                s, github_token_configured=_gh_token_configured, hub_catalog_files=hub_catalog_files,
            ),
            "conflict": skill_conflict_status(s, skills),
            "load_error": runtime_states.get(s.name, {}).get("load_error", s.load_error),
            "desired_live": runtime_states.get(s.name, {}).get("desired_live", False),
            "live_loaded": runtime_states.get(s.name, {}).get("live_loaded", False),
            "live_reason": runtime_states.get(s.name, {}).get("reason", "not_extension"),
            **extension_process_receipt(runtime_states.get(s.name)),
            "health_regressed": bool((health or {}).get("regressed")),
            "last_known_good": (health or {}).get("last_known_good"),
            "health_observations": (health or {}).get("observations", {}),
            "dispatch_live": bool(
                _live_tool_count(s.name)
                or _live_route_count(s.name)
                or _live_ws_count(s.name)
            ),
            "review_findings": list(s.review.findings or []),
            "skill_review": skill_review_ui_projection(drive_root, s.name),
            # An extension row reuses the grant status its runtime state computed
            # above (same function, same drive_root and skill): one settings
            # read per skill, not two.
            "grants": (
                runtime_states[s.name]["grant_status"]
                if s.name in runtime_states
                else grant_status_for_skill(drive_root, s)
            ),
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
    # The hub hint hashes payload bytes: off the event loop, like the index.
    review_fields = await asyncio.to_thread(
        lambda: _review_fields(loaded, hub_catalog_files=display_catalog_files()),
    )
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
            **review_fields,
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
        state = await asyncio.to_thread(
            runtime_state_for_skill_name, skill, drive_root, repo_path=repo_path,
        )
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
            return child_result
        except Exception as exc:
            log.exception("extension child dispatch failure: %s", mount)
            return json_error(f"{type(exc).__name__}: {exc}", 502)
    handler = spec.get("handler")
    if not callable(handler):
        return json_error("registered handler is not callable")
    try:
        from ouroboros.extension_process_runner import disclose_inprocess_extension_dispatch

        # A ledger append under the money lock for an extension holding a funded
        # provider key: a lock wait, so it runs off the event loop.
        await asyncio.to_thread(
            disclose_inprocess_extension_dispatch,
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
    """Toggle through the shared owner of grant/review/dependency preconditions."""
    skill_name = str(request.path_params.get("skill") or "").strip()
    if not skill_name:
        return json_error("missing skill name", 400)
    body = await request_json_or(request, {}, exceptions=(Exception,))
    if not isinstance(body, dict):
        return json_error("request body must be a JSON object", 400)
    sentinel = object()
    enabled = coerce_bool(body.get("enabled"), default=sentinel)
    if enabled is sentinel:
        return json_error("'enabled' must be a boolean", 400)
    payload = await _run_owner_skill_action(request, skill_name, "enable" if enabled else "disable", body)
    if payload.get("error"):
        return JSONResponse(payload, status_code=int(payload.get("status_code") or 400))
    return JSONResponse({
        "skill": payload.get("skill", skill_name), "enabled": payload.get("enabled", False),
        "review_status": payload.get("review_status"), "review_stale": payload.get("review_stale"),
        "review_gate": payload.get("review_gate"), "executable_review": payload.get("executable_review"),
        "grants": payload.get("grants", {}), "extension_action": payload.get("extension_action"),
        "extension_reason": payload.get("extension_reason"), **extension_process_receipt(payload),
    })


class _ApiReviewCtx:
    """Minimal ToolContext-compatible carrier for HTTP-triggered review."""

    def __init__(self, drive_root: pathlib.Path, repo_dir: pathlib.Path) -> None:
        self.drive_root = drive_root
        self.repo_dir = repo_dir
        self.task_id = "api_skill_review"
        self.task_lifecycle_bound = False
        self.current_chat_id = 0
        self.pending_events: list = []
        self.emit_progress_fn = None
        self.event_queue = None  # _emit_usage_event falls back to pending_events
        self.messages: list = []


async def _run_owner_skill_action(request: Request, skill_name: str, action: str, body: dict) -> dict:
    from ouroboros.config import get_skills_repo_path
    from ouroboros.skill_lifecycle_actions import run_skill_action

    ctx = _ApiReviewCtx(_request_drive_root(request), _request_repo_dir(request))
    ctx.task_id = ""  # A UI lifecycle action is not a managed task.
    ctx._skill_owner_client_host = str(getattr(getattr(request, "client", None), "host", "") or "")
    payload = await run_blocking_preserving_cancellation(
        run_skill_action, ctx, skill_name, action,
        expected_content_hash=str(body.get("expected_content_hash") or ""),
        items=_grant_items_from_body(body) if action == "grant" else None,
        payload_root=str(body.get("payload_root") or ""),
        repo_path=get_skills_repo_path(), _owner_actor="owner_ui",
        log_label=f"skill {action} lifecycle operation",
    )
    _broadcast_extension_lifecycle(request, skill_name, payload.get("extension_action"), payload.get("extension_reason"))
    return payload


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
    """Owner-requested, preflight-floored attestation through the common lifecycle."""
    skill_name = str(request.path_params.get("skill") or "").strip()
    if not skill_name:
        return json_error("missing skill name", 400)
    body = await request_json_or(request, {}, exceptions=(Exception,))
    if not isinstance(body, dict):
        return json_error("request body must be a JSON object", 400)
    payload = await _run_owner_skill_action(request, skill_name, "attest", body)
    return JSONResponse(payload, status_code=200 if payload.get("ok") else int(payload.get("status_code") or 409))


async def api_skill_lifecycle_queue(_request: Request) -> JSONResponse:
    """GET /api/skills/lifecycle-queue — recent mutating skill operations.

    The queue snapshot only: polled every second while a lifecycle action is
    pending, so it heals nothing (the review-job owners do)."""
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
    """Grant only current manifest items through the shared lifecycle owner."""
    skill_name = str(request.path_params.get("skill") or "").strip()
    if not skill_name:
        return json_error("missing skill name", 400)
    body = await request_json_or(request, {}, exceptions=(Exception,))
    if not isinstance(body, dict):
        return json_error("request body must be a JSON object", 400)
    payload = await _run_owner_skill_action(request, skill_name, "grant", body)
    return JSONResponse(payload, status_code=int(payload.get("status_code") or 200))


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
    # Reconcile can flip grants/load state, so refresh schedule readiness now —
    # off the event loop: the resync waits for the supervisor queue lock.
    try:
        from supervisor.queue import resync_skill_schedules

        await asyncio.to_thread(resync_skill_schedules, drive_root)
    except Exception:
        log.debug("api_skill_reconcile schedule sync failed", exc_info=True)
    return JSONResponse(extension_reconcile_receipt(skill_name, state))


async def api_skill_delete(request: Request) -> JSONResponse:
    """Delete an explicitly selected local payload through its existing owner."""
    skill_name = _sanitize_skill_name(str(request.path_params.get("skill") or "").strip())
    if not skill_name or skill_name == "_unnamed":
        return json_error("missing skill name", 400)
    body = await request_json_or(request, {}, exceptions=(Exception,))
    if not isinstance(body, dict):
        return json_error("request body must be a JSON object", 400)
    payload = await _run_owner_skill_action(request, skill_name, "delete", body)
    return JSONResponse(payload, status_code=int(payload.get("status_code") or 200))


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

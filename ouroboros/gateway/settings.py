"""Settings, onboarding, and Claude-runtime gateway endpoints."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import pathlib
import socket
import sys
from typing import Any, Dict, Optional

from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, Response

from ouroboros.config import (
    DATA_DIR,
    SETTINGS_DEFAULTS as _SETTINGS_DEFAULTS,
    apply_settings_to_env as _apply_settings_to_env,
    load_settings,
)
from ouroboros.config import ENDPOINT_AUTHORED_SETTINGS as _ENDPOINT_AUTHORED_SETTINGS
from ouroboros.config import ENDPOINT_WRITTEN_SETTINGS as _ENDPOINT_WRITTEN_SETTINGS, ENDPOINT_WRITERS as _ENDPOINT_WRITERS
from ouroboros.gateway._helpers import json_error, json_exception, request_drive_root
from ouroboros.gateway.owner_settings import (
    CommitBoundary,
    SettingsDocumentBusy,
    SettingsLockUnavailable,
    _CONTEXT_MODE_KEYS,
    _owner_audit,
    _owner_read_settings_raw,
    _owner_update_settings,
    _owner_write_settings,
    settings_document_mutation,
    owner_write_guard,
    post_commit_failure_response,
    settings_document_digest,
    unsaved_error,
)
from ouroboros.onboarding_wizard import build_onboarding_html
from ouroboros.platform_layer import is_container_env
from ouroboros.provider_models import (
    MINIMAX_REGION_ENDPOINTS, ZAI_PLAN_ENDPOINTS, resolve_minimax_base_url, resolve_zai_base_url,
)
from ouroboros.secret_masking import (
    MCP_RESPONSE_ONLY_FIELDS,
    is_custom_secret_setting_key,
    looks_masked_mcp_secret,
    looks_masked_settings_secret,
    mask_prefixed_secret,
    mask_mcp_url,
    rehydrate_mcp_url,
    mask_settings_secret,
)
from ouroboros.server_runtime import (
    apply_runtime_provider_defaults,
    classify_runtime_provider_change,
    has_startup_ready_provider,
)
from ouroboros.settings_setup_contract import (
    BUDGET_SETTING_KEYS,
    SECRET_SETTING_KEYS,
    build_setup_contract,
    parse_budget_setting,
)
log = logging.getLogger(__name__)
DEFAULT_PORT = int(os.environ.get("OUROBOROS_SERVER_PORT", "8765"))


def _trust_nonlocal_bind_without_password_enabled() -> bool:
    raw = os.environ.get("OUROBOROS_TRUST_NONLOCAL_BIND_WITHOUT_PASSWORD", "")
    return str(raw or "").strip().lower() in {"1", "true", "yes", "on"}


def _build_network_meta(bind_host: str, bind_port: int) -> dict:
    """Build /api/settings network metadata."""
    from ouroboros.server_auth import get_network_auth_startup_warning, is_loopback_host
    # Strip IPv6 brackets before loopback classification.
    unbracketed = bind_host[1:-1] if bind_host.startswith("[") and bind_host.endswith("]") else bind_host
    loopback = is_loopback_host(unbracketed)
    if loopback:
        return {
            "bind_host": bind_host,
            "bind_port": bind_port,
            "lan_ip": "",
            "reachability": "loopback_only",
            "recommended_url": "",
            "warning": "Server is bound to localhost — not accessible from other devices.",
        }
    wildcard = bind_host in ("0.0.0.0", "")
    if wildcard:
        lan_ip = ""
        if not is_container_env():
            # The LAN IP via the UDP socket trick; no packet is sent.
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                    s.connect(("192.0.2.1", 80))  # RFC 5737 TEST-NET-1, no packet sent
                    lan_ip = s.getsockname()[0]
            except OSError:
                lan_ip = ""
    elif bind_host in ("::", "[::]"):
        # AF_INET startup cannot advertise an IPv6 wildcard LAN IP reliably.
        lan_ip = ""
    else:
        # Use unbracketed form so URL construction can re-bracket IPv6 uniformly.
        lan_ip = unbracketed

    auth_warning = get_network_auth_startup_warning(bind_host) or ""
    if lan_ip:
        host_in_url = f"[{lan_ip}]" if ":" in lan_ip else lan_ip
        reachability = "lan_reachable"
        recommended_url = f"http://{host_in_url}:{bind_port}"
        warning = auth_warning
    else:
        reachability = "host_ip_unknown"
        recommended_url = f"http://your-host-ip:{bind_port}"
        warning = " ".join(
            part for part in [
                "Could not detect LAN IP automatically." if wildcard else "",
                auth_warning,
            ]
            if part
        )
    return {
        "bind_host": bind_host,
        "bind_port": bind_port,
        "lan_ip": lan_ip,
        "reachability": reachability,
        "recommended_url": recommended_url,
        "warning": warning,
    }


def _mask_mcp_servers_payload(servers: Any) -> list:
    if not isinstance(servers, list):
        return []
    try:
        from ouroboros.mcp_client import canonical_server_id as _mcp_canonical_id
    except Exception:
        _mcp_canonical_id = lambda value: str(value or "").strip()  # type: ignore[assignment]
    out = []
    for entry in servers:
        if not isinstance(entry, dict):
            continue
        clone = dict(entry)
        if clone.get("id"):
            clone["id"] = _mcp_canonical_id(clone.get("id"))
        if "url" in clone:
            clone["url"] = mask_mcp_url(clone["url"])
        token = str(clone.get("auth_token") or "")
        if token:
            clone["auth_token"] = mask_prefixed_secret(token, visible_chars=8)
            clone["auth_configured"] = True
        else:
            clone["auth_token"] = ""
            clone["auth_configured"] = False
        out.append(clone)
    return out


def _build_policy_state(settings: Dict[str, Any]) -> dict:
    """Project configured versus process-effective owner policy for Settings UI.

    Persisted values are the pending choices.  Runtime access is boot-bound;
    Supervisor and Review are hot-reloaded for the next task through the
    existing settings path.  The projection deliberately carries no authority
    and writes no second state record.
    """
    from ouroboros import config as _config
    from ouroboros.review_model_routes import get_review_enforcement

    configured_access = _config.normalize_runtime_mode(
        settings.get("OUROBOROS_RUNTIME_MODE"))
    effective_access = _config.get_runtime_mode()
    configured_supervisor = _config.normalize_safety_mode(
        settings.get("OUROBOROS_SAFETY_MODE"))
    effective_supervisor = _config.get_safety_mode()
    configured_review = str(
        settings.get("OUROBOROS_REVIEW_ENFORCEMENT") or "advisory").strip().lower()
    effective_review = get_review_enforcement()
    running_task_snapshot = bool(_has_started_agent_tasks())
    return {
        "access": {
            "configured": configured_access,
            "effective": effective_access,
            "current_process": effective_access,
            "next_task": configured_access,
            "restart_required": configured_access != effective_access,
            "applies": "restart",
        },
        "supervisor": {
            "configured": configured_supervisor,
            "effective": effective_supervisor,
            "current_process": effective_supervisor,
            "next_task": configured_supervisor,
            "pending": configured_supervisor != effective_supervisor or running_task_snapshot,
            "applies": "next_task",
            "active_task_snapshot": running_task_snapshot,
        },
        "review": {
            "configured": configured_review if configured_review in {"advisory", "blocking"} else "advisory",
            "effective": effective_review,
            "current_process": effective_review,
            "next_task": configured_review if configured_review in {"advisory", "blocking"} else "advisory",
            "pending": configured_review != effective_review or running_task_snapshot,
            "applies": "next_task",
            "active_task_snapshot": running_task_snapshot,
        },
        "running_task_snapshot": running_task_snapshot,
    }


def _build_restart_state(settings: Dict[str, Any]) -> dict:
    """Compare saved intent with component-owned inputs, never os.environ."""
    from ouroboros.config import get_runtime_mode, normalize_runtime_mode
    from ouroboros.local_model import get_manager, local_model_settings
    from ouroboros.server_process import applied_restart_settings, applied_server_host_source

    applied = applied_restart_settings()
    desired = {key: settings.get(key, _SETTINGS_DEFAULTS.get(key, ""))
               for key in _RESTART_REQUIRED_KEYS if key not in local_model_settings({})}
    applied["OUROBOROS_RUNTIME_MODE"] = get_runtime_mode()
    desired["OUROBOROS_RUNTIME_MODE"] = normalize_runtime_mode(settings.get("OUROBOROS_RUNTIME_MODE"))
    pending = []
    for key, value in desired.items():
        if key not in applied:
            continue
        actual = applied[key]
        if key == "OUROBOROS_SKILLS_REPO_PATH":
            value = str(pathlib.Path(str(value).strip()).expanduser()) if str(value).strip() else ""
            actual = str(pathlib.Path(str(actual).strip()).expanduser()) if str(actual).strip() else ""
        if str(value).strip() != str(actual).strip():
            pending.append(key)
    unknown = sorted(set(desired) - set(applied))
    host_key = "OUROBOROS_SERVER_HOST"
    host_source = applied_server_host_source(DATA_DIR)
    source_unknown = []
    host_summary = ""
    if host_key in pending and host_source != "settings":
        pending.remove(host_key)
        if host_source in {"environment", "cli"}:
            host_summary = " Saved server host differs from the running listener; launch configuration overrides this setting."
        else:
            source_unknown.append(host_key)
            host_summary = (" Saved server host differs from the running listener. This launcher did not report "
                            "whether a launch override controls the next start; Restart may apply the saved host.")
    local = get_manager().settings_application(settings)
    summary = f"Restart Ouroboros to apply {len(pending)} saved setting(s)." if pending else ""
    if unknown:
        summary += f" Application state is not reported for {len(unknown)} runtime setting(s)."
    summary += host_summary
    return {"restart_required": bool(pending), "restart_keys": sorted(pending),
            "restart_source_unknown_keys": source_unknown, "unknown_keys": unknown,
            "local_model": local, "summary": summary.strip()}


def _rehydrate_mcp_servers_payload(incoming: Any, current: Any) -> list:
    """Restore masked MCP tokens/URLs from the ONE saved server with the same identity.

    Identity is the loader's (``id``, ``slug`` or ``name``). A masked value whose
    identity has several saved or incoming claimants raises
    ``MCPSecretIdentityAmbiguous`` before anything is written: an equal token
    does not make two servers the same, so a credential never moves between them.
    """
    if not isinstance(incoming, list):
        return []
    from ouroboros.mcp_client import canonical_server_id, raw_server_id

    current_by_id: Dict[str, list] = {}
    for entry in current if isinstance(current, list) else []:
        if raw_server_id(entry):
            current_by_id.setdefault(raw_server_id(entry), []).append(entry)
    incoming_ids = [raw_server_id(entry) for entry in incoming if isinstance(entry, dict)]
    out = []
    for entry in incoming:
        if not isinstance(entry, dict):
            continue
        clone = {key: value for key, value in entry.items() if key not in MCP_RESPONSE_ONLY_FIELDS}
        if clone.get("id"):
            clone["id"] = canonical_server_id(clone.get("id"))
        server_id = raw_server_id(clone)
        token = str(clone.get("auth_token") or "")
        masked = looks_masked_mcp_secret(token) or (
            "url" in clone and rehydrate_mcp_url(clone["url"], "") != str(clone["url"] or ""))
        matches = current_by_id.get(server_id, []) if server_id else []
        if masked and (len(matches) > 1 or incoming_ids.count(server_id) > 1):
            raise MCPSecretIdentityAmbiguous(server_id)
        existing = matches[0] if len(matches) == 1 else {}
        if "url" in clone:
            clone["url"] = rehydrate_mcp_url(clone["url"], existing.get("url"))
        if looks_masked_mcp_secret(token):
            clone["auth_token"] = str(existing.get("auth_token") or "")
        out.append(clone)
    return out


class MCPSecretIdentityAmbiguous(ValueError):
    """A masked MCP secret whose server identity is claimed more than once."""

    code = "MCP_ID_AMBIGUOUS_SECRET"

    def __init__(self, server_id: str) -> None:
        super().__init__(
            f"{self.code}: several MCP servers resolve to server id {server_id!r}, so its saved "
            "credential cannot be matched to one of them. Give each server a distinct Server ID "
            "(or re-enter its token/URL) and save again; nothing was saved."
        )


from ouroboros.settings_scales import (
    IMMEDIATE_SETTINGS as _IMMEDIATE_KEYS,
    RESTART_REQUIRED_SETTINGS as _RESTART_REQUIRED_KEYS,
)


def _effect_buckets(all_changed: list) -> tuple:
    """Split changed keys into the honest effect buckets for the save response.

    Every key that reaches here applies at SOME point, so both buckets are
    truthful claims (#285). A key a release retired cannot reach here at all:
    the merge only walks SETTINGS_DEFAULTS, and load_settings strips retired
    keys off disk — the RC auditor is what tells an upgrading install its
    stored value is gone (scripts/rc_audit.py "retired-setting").
    """
    immediate_changed = [k for k in all_changed if k in _IMMEDIATE_KEYS]
    next_task_changed = [
        k for k in all_changed
        if k not in _IMMEDIATE_KEYS and k not in _RESTART_REQUIRED_KEYS and k != "OUROBOROS_RUNTIME_MODE"
    ]
    return immediate_changed, next_task_changed


def _merge_settings_payload(current: Dict[str, Any], body: Dict[str, Any]) -> Dict[str, Any]:
    merged = {k: v for k, v in current.items()}
    from ouroboros.config import get_runtime_mode
    from ouroboros.runtime_mode_policy import runtime_mode_at_least

    skipped = {"OUROBOROS_CONTEXT_MODE_AUTO_LOW"} | _ENDPOINT_AUTHORED_SETTINGS | _ENDPOINT_WRITTEN_SETTINGS
    if not runtime_mode_at_least(get_runtime_mode(), "cyber_pro"):
        skipped |= {"OUROBOROS_CONTEXT_MODE", "OUROBOROS_RUNTIME_MODE", "OUROBOROS_AUTO_GRANT_REVIEWED_SKILLS", "OUROBOROS_SAFETY_MODE"}
    # Cyber authors controls through the same writer. Install receipts remain facts;
    # the retired auto-Low marker is derived from an explicit context choice below.
    for key in _SETTINGS_DEFAULTS:
        if key in skipped:
            continue
        if key not in body:
            continue
        # A placeholder means "field untouched", never a new secret — and never a
        # credential worth persisting even when nothing is stored yet. Clearing
        # stays explicit: the UI sends "" for its Clear action.
        if key in SECRET_SETTING_KEYS and looks_masked_settings_secret(key, body[key]):
            continue
        merged[key] = body[key]
    if "OUROBOROS_CONTEXT_MODE" in body and "OUROBOROS_CONTEXT_MODE" not in skipped:
        from ouroboros.config import normalize_context_mode
        merged["OUROBOROS_CONTEXT_MODE"] = normalize_context_mode(body["OUROBOROS_CONTEXT_MODE"])
        merged["OUROBOROS_CONTEXT_MODE_AUTO_LOW"] = "false"
    for key, value in body.items():
        text_key = str(key or "").strip().upper()
        if text_key in _SETTINGS_DEFAULTS or text_key == "OUROBOROS_RUNTIME_MODE":
            continue
        if not is_custom_secret_setting_key(
            text_key, known_setting_keys=_SETTINGS_DEFAULTS
        ):
            continue
        if looks_masked_settings_secret(text_key, value):
            continue
        merged[text_key] = value
    return merged


def _current_bind_host(request: Request) -> str:
    return str(getattr(getattr(request.app, "state", None), "bind_host", "") or "")


def _port_file(request: Request) -> pathlib.Path:
    configured = getattr(getattr(request.app, "state", None), "port_file", None)
    return pathlib.Path(configured) if configured is not None else pathlib.Path(DATA_DIR) / "state" / "server_port"


def _default_port(request: Request) -> int:
    return int(getattr(getattr(request.app, "state", None), "default_port", DEFAULT_PORT) or DEFAULT_PORT)


def _start_supervisor_if_needed_for_request(request: Request, settings: dict) -> bool:
    callback = getattr(getattr(request.app, "state", None), "start_supervisor_if_needed", None)
    return bool(callback(settings)) if callable(callback) else False


async def _json_body_or_empty(request: Request) -> Any:
    try:
        return await request.json()
    except Exception:
        return {}


def _has_running_agent_tasks() -> bool:
    # Known pre-existing residual: PENDING/RUNNING are read without the queue
    # lock, so a task mid-handoff (removed from PENDING, not yet in RUNNING)
    # can be invisible to one snapshot. The context-mode guard tolerates it —
    # the read races the supervisor thread with or without any settings lock.
    try:
        from supervisor.workers import PENDING, RUNNING
        from supervisor.active_activity import get_direct_activity_registry
        if PENDING or RUNNING:
            return True
        return bool(get_direct_activity_registry().snapshot())
    except Exception:
        return False


def _has_started_agent_tasks() -> bool:
    """STARTED tasks only — the ones the snapshot boundary actually binds.

    A queued-but-unstarted task re-reads settings in ``handle_task``, so warning
    that it "keeps the previous configuration" would be false; ``PENDING`` is
    deliberately excluded (unlike ``_has_running_agent_tasks``, whose callers
    gate on any outstanding work). Registry inspection includes native turns
    without constructing an actor merely to answer a status question."""
    try:
        import supervisor.workers as _workers
        if _workers.RUNNING:
            return True
        from supervisor.active_activity import get_direct_activity_registry

        return bool(get_direct_activity_registry().snapshot())
    except Exception:
        return False


async def _run_settings_writer(fn: Any, context: Any, body: Any) -> JSONResponse:
    """Run one settings WRITER endpoint body off the event loop, bounded.

    Every writer serializes on the bounded in-process document lock
    (``settings_document_mutation``); its typed refusal answers the same
    503 ``settings_busy`` on every endpoint — the generic save, the four
    single-decision endpoints and onboarding alike — never an untyped 500
    from one of them while another answers honestly.

    The INITIATING writer (onboarding completion included) is bounded by the same
    contract the lock enforces on later writers (``OUROBOROS_SETTINGS_DOCUMENT_LOCK_TIMEOUT_SEC``):
    one lock wait plus one held episode, each within that bound. A body wedged
    in its post-commit effects (extension reload, remote configuration) is
    abandoned to its uncancellable worker thread and the Save answers a typed
    503 ``settings_save_timeout`` with ``saved: null`` — whether the bytes
    landed is genuinely unknown then, so neither ``true`` nor ``false`` would
    be honest. Later writers keep getting ``settings_busy`` meanwhile.
    """
    from ouroboros.config import get_settings_document_lock_timeout_sec

    bound_sec = get_settings_document_lock_timeout_sec()
    # The body runs in its own task so a timeout of the WAIT can be told apart from a
    # TimeoutError raised BY the body (asyncio.TimeoutError is the builtin on 3.11+).
    episode = asyncio.ensure_future(asyncio.to_thread(fn, context, body))
    try:
        return await asyncio.wait_for(asyncio.shield(episode), timeout=2 * bound_sec)
    except SettingsDocumentBusy as exc:
        return unsaved_error(str(exc), 503, code="settings_busy")
    except (asyncio.TimeoutError, TimeoutError):
        if episode.done():
            # The body FINISHED: it raised (re-raised here as an ordinary failure), or its
            # result landed in the tick the timer fired — ``wait_for`` abandons the shield one
            # tick before reading it — and a finished save is answered, not thrown away.
            return episode.result()
        log.warning(
            "Settings writer %s did not answer within %ss; its thread keeps running",
            getattr(fn, "__name__", fn), 2 * bound_sec,
        )
        return json_error(
            f"the settings save is still running in the server after {2 * bound_sec}s "
            "and was left to finish on its own; reload Settings to see what landed",
            503, code="settings_save_timeout", saved=None,
        )


@owner_write_guard
async def api_owner_runtime_mode(request: Request) -> JSONResponse:
    """Persist the owner-selected runtime mode for the next boot."""
    body = await _json_body_or_empty(request)
    # Off the event loop, under the document lock (held inside): a slow
    # generic save must not be able to freeze the loop THROUGH this
    # endpoint's synchronous lock acquisition.
    return await _run_settings_writer(_api_owner_runtime_mode_sync, request, body)


def _api_owner_runtime_mode_sync(request: Request, body: Any) -> JSONResponse:
    from ouroboros import config as _config

    raw_mode = str((body or {}).get("mode") or "").strip().lower()
    if raw_mode not in set(_config.VALID_RUNTIME_MODES):
        return unsaved_error("'mode' must be one of: light, advanced, pro, cyber_pro", 400)
    # The digest is taken BEFORE the read that decides, so a write landing between the
    # two is refused rather than silently reverted by this request's write.
    digest = settings_document_digest()
    old_settings = _owner_read_settings_raw()
    previous_mode = _config.normalize_runtime_mode(old_settings.get("OUROBOROS_RUNTIME_MODE"))
    active_mode = _config.get_runtime_mode()
    next_mode = _config.normalize_runtime_mode(raw_mode)
    restart_required = active_mode != next_mode
    if next_mode != previous_mode:
        # A no-change POST must not rewrite settings.json: the rewrite raced a
        # concurrent generic save (last-writer-wins over a stale read) for zero
        # information gain. The audit and the response stay identical either way.
        def _set_runtime_mode(current: Dict[str, Any]) -> Dict[str, Any]:
            current["OUROBOROS_RUNTIME_MODE"] = next_mode
            return current

        # Under the seam-wide document lock, because a threaded generic save may
        # be mid read-merge-write on the same document. The transform's own read
        # happens inside the settings lock, so there is no second stale read to
        # refresh here; the digest keeps this request's PRE-lock decision bound to
        # the document that decision was taken from.
        with settings_document_mutation():
            _owner_update_settings(_set_runtime_mode, digest)
    _owner_audit(
        request,
        "runtime_mode",
        {
            "runtime_mode": next_mode,
            "previous_runtime_mode": previous_mode,
            "active_runtime_mode": active_mode,
            "restart_required": restart_required,
        },
    )
    return JSONResponse({
        "ok": True,
        "runtime_mode": next_mode,
        "restart_required": restart_required,
    })


@owner_write_guard
async def api_owner_auto_grant(request: Request) -> JSONResponse:
    """Persist the owner auto-grant toggle outside generic settings writes."""
    body = await _json_body_or_empty(request)
    # Off the event loop, under the document lock (held inside): a slow
    # generic save must not be able to freeze the loop THROUGH this
    # endpoint's synchronous lock acquisition.
    return await _run_settings_writer(_api_owner_auto_grant_sync, request, body)


def _api_owner_auto_grant_sync(request: Request, body: Any) -> JSONResponse:
    if not isinstance(body, dict) or not isinstance(body.get("enabled"), bool):
        return unsaved_error("'enabled' must be a boolean", 400)
    enabled = bool(body.get("enabled"))
    value = "true" if enabled else "false"

    # No digest: this endpoint decides nothing from the stored document — the body
    # carries the whole decision — so refusing a concurrent unrelated write would
    # cost the owner a retry and buy nothing.
    def _set_auto_grant(current: Dict[str, Any]) -> Dict[str, Any]:
        current["OUROBOROS_AUTO_GRANT_REVIEWED_SKILLS"] = value
        return current

    with settings_document_mutation():
        _owner_update_settings(_set_auto_grant)
        # Projected under the SAME lock as the commit: released first, two
        # writers can commit A->B and project B->A, stranding the live
        # environment on the loser's value.
        os.environ["OUROBOROS_AUTO_GRANT_REVIEWED_SKILLS"] = value
    _owner_audit(request, "auto_grant", {"enabled": enabled})
    return JSONResponse({"ok": True, "enabled": enabled})


def _active_main_route(
    settings: Dict[str, Any],
    *,
    model_override: str = "",
    use_local_override: Optional[bool] = None,
) -> Dict[str, Any]:
    """(provider, model, base_url, use_local) for the active main model route.

    ``model_override`` / ``use_local_override`` let the task loop probe the ACTUAL
    active route at point-of-use (CW2) — a per-task ``switch_model`` / model override
    or a mid-loop local-route change — rather than only the settings-derived route."""
    from ouroboros import config as _config
    from ouroboros.provider_models import provider_for_model

    model = str(model_override or settings.get("OUROBOROS_MODEL") or _config.SETTINGS_DEFAULTS.get("OUROBOROS_MODEL") or "").strip()
    provider = provider_for_model(model)
    # The settings key a provider's base URL resolves through.
    base_url_key = {"openai": "OPENAI_BASE_URL", "openai-compatible": "OPENAI_COMPATIBLE_BASE_URL",
                    "cloudru": "CLOUDRU_FOUNDATION_MODELS_BASE_URL", "gigachat": "GIGACHAT_BASE_URL"}.get(provider)
    if provider == "minimax":
        base_url = resolve_minimax_base_url(settings.get("MINIMAX_REGION") or "")
    elif provider == "zai":
        base_url = resolve_zai_base_url(settings.get("ZAI_PLAN") or "")
    else:
        base_url = str(settings.get(base_url_key) or "") if base_url_key else ""
    # CW7 (v6.34.0): honour the USE_LOCAL_MAIN routing setting — a local-routed main
    # lane must report provider='local' so the Max gate consults the local n_ctx
    # (Capability Evidence local-health) instead of the remote OUROBOROS_MODEL metadata.
    use_local_main = str(settings.get("USE_LOCAL_MAIN") or "").strip().lower() in ("1", "true", "yes", "on")
    use_local = use_local_main or model.endswith(" (local)") or provider == "local"
    if use_local_override is not None:
        use_local = bool(use_local_override)
    if use_local:
        provider = "local"
    return {"provider": provider, "model": model, "base_url": base_url, "use_local": use_local}


def _unrecognised_review_models(models: Any) -> list:
    """Review-slot model ids the provider catalog does not know (evidence-based).

    An OpenRouter-routed id that is ABSENT from a SUCCESSFULLY fetched OpenRouter
    catalog is reported loudly: a truncated slot value (e.g. ``-5``) otherwise looks
    valid on save and only surfaces later as three waves of ``400 <id> is not a valid
    model ID``, which destroys the review quorum. Nothing is rejected or rewritten —
    the fetch may be unavailable, so this is a WARNING, never a save gate."""
    try:
        from ouroboros.llm import LLMClient
        from ouroboros.provider_models import provider_for_model

        candidates = [str(m or "").strip() for m in (models or []) if str(m or "").strip()]
        openrouter = [m for m in candidates if provider_for_model(m) == "openrouter"]
        if not openrouter:
            return []
        LLMClient.openrouter_context_length(openrouter[0], allow_fetch=True)
        if not getattr(LLMClient, "_CAPABILITIES_FETCH_OK", False):
            return []  # no authoritative catalog -> cannot claim anything is unknown
        known = getattr(LLMClient, "_CONTEXT_LENGTH_CACHE", {}) or {}
        return [m for m in openrouter if m not in known]
    except Exception:
        return []


REVIEW_LANES_KEY = "OUROBOROS_REVIEWER_SLOTS"
ALLOW_EMPTY_REVIEW_POOL = "allow_empty_review_pool"


def _catalog_document(raw: Any) -> Dict[str, Any]:
    """The stored catalog as a plain object for display and price facts ({} when
    unreadable); whether it is VALID is package A's judgement (``review_pool_state``)."""
    try:
        payload = json.loads(raw) if isinstance(raw, str) and raw.strip() else raw
    except ValueError:
        payload = None
    payload = payload if isinstance(payload, dict) else {}
    items = payload.get("items") if isinstance(payload.get("items"), list) else []
    return {**payload, "items": [item for item in items if isinstance(item, dict)]}


def _candidate_pool_api_models(settings: Dict[str, Any]) -> list:
    """Provider model ids of the API rows marked Reviewer (a session target is a route spec)."""
    rows = _catalog_document(settings.get("OUROBOROS_SUBAGENTS"))["items"]
    return [str(row["route"].get("target_id") or "") for row in rows
            if row.get("review_eligible") is True and isinstance(row.get("route"), dict)
            and row["route"].get("kind") == "api_model"]


def review_pool_save_judgement(raw: Any, stored: Dict[str, Any], *, allow_empty: bool) -> str:
    """The empty-pool rule of a catalog save ('' = acceptable). It judges only a save that
    CHANGES the catalog a Settings read showed — the stored one, else the unsaved candidate
    offered in its place, whose pool was already the install's — or retires stored review
    lanes, so no pool empties silently; ``allow_empty_review_pool`` confirms one."""
    from ouroboros.configured_subagents import (
        SUBAGENTS_SETTING, configured_subagents_dict, normalize_configured_subagents,
        resolve_settings_subagent_candidate,
    )

    def canonical(value: Any) -> str:
        try:
            return normalize_configured_subagents(value)[1]
        except ValueError:
            return ""  # nothing readable: this save authors the catalog

    shown = stored.get(SUBAGENTS_SETTING)
    if shown in (None, ""):
        config = resolve_settings_subagent_candidate(apply_runtime_provider_defaults(dict(stored))[0])[0].config
        shown = configured_subagents_dict(config) if config is not None else None
    posted = canonical(raw)
    if allow_empty or (posted == canonical(shown) and not str(stored.get(REVIEW_LANES_KEY) or "").strip()):
        return ""
    from ouroboros.reviewer_slot_config import review_pool_save_error

    return review_pool_save_error(posted, allow_empty=False)


def _review_pool_costs(items: list, snapshot: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """The price of every catalog row. A session, or a model call through a subscription,
    uses a seat and time; an API row's ``usd_per_review`` is ``review_row_call_usd``: one
    full call of the row, the number ``## Review`` shows for its seat (a reading reviewer
    makes several); a route without a tariff stays unknown — there are no price tables."""
    from ouroboros.model_slots import resolve_processing_preference
    from ouroboros.provider_models import provider_for_model
    from ouroboros.tools.review_helpers import review_row_call_usd

    costs: Dict[str, Dict[str, Any]] = {}
    for item in items:
        row_id, route = str(item.get("subagent_id") or ""), item.get("route") or {}
        target = str(route.get("target_id") or "")
        seat = route.get("kind") == "agent_session" or provider_for_model(target) == "claudexor"
        usd = None if seat or not target else review_row_call_usd({
            "slot_id": row_id, "model": target, "profile_id": str(route.get("credential_profile_id") or ""),
            "processing_preference": resolve_processing_preference(
                override=item.get("processing_preference") or None, settings=snapshot)})
        costs[row_id] = ({"usd_per_review": None, "basis": "subscription_seat" if seat else "unknown"}
                         if usd is None else {"usd_per_review": float(usd), "basis": "route_tariff"})
    return costs


def review_pool_rows(items: list, slots: list, handles: Dict[str, str],
                     last_executions: Dict[str, Any], costs: Dict[str, Dict[str, Any]]) -> tuple:
    """(pool, excluded) of ``GET /api/review-pool``: pool slots in catalog order, each joined with its row's facts."""
    from ouroboros.route_spec import ROUTE_KIND_AGENT_SESSION, ROUTE_KIND_API_MODEL, RouteSpec, model_named_effort

    by_id = {str(item.get("subagent_id") or ""): item for item in items}
    pool = []
    for slot in slots:
        row_id = str(getattr(slot, "subagent_id", "") or getattr(slot, "slot_id", "") or "")
        item = by_id.get(row_id, {})
        route = item.get("route") or {}
        target = str(route.get("target_id") or "")
        session = route.get("kind") == ROUTE_KIND_AGENT_SESSION
        named = model_named_effort(RouteSpec(ROUTE_KIND_AGENT_SESSION if session else ROUTE_KIND_API_MODEL, target))
        pool.append({
            "subagent_id": row_id, "handle": handles.get(row_id, row_id),
            "route": {"kind": str(route.get("kind") or ""), "target_id": target,
                      "credential_profile_id": str(route.get("credential_profile_id") or "")},
            "effort": str(getattr(slot, "effort", "") or ""),
            # Who decides the level with no order: the model name, the owner's pin, or Auto (the range's top).
            "effort_source": "model_name" if named else ("pin" if item.get("effort") else "auto"),
            "delivery": "session" if session else ("packet" if item.get("delivery") == "packet" else "native"),
            "processing_preference": str(item.get("processing_preference") or ""),
            "access": str(item.get("access") or ("full" if session else "")),
            "enabled": item.get("enabled") is not False, "review_eligible": True,
            "minted_from": str(item.get("minted_from") or ""),
            "cost": costs.get(row_id) or {"usd_per_review": None, "basis": "unknown"}, "last_execution": last_executions.get(row_id),
        })
    pooled = {row["subagent_id"] for row in pool}
    excluded = [{"subagent_id": row_id, "reason": "row_disabled"} for row_id, item in by_id.items()
                if item.get("review_eligible") is True and item.get("enabled") is False and row_id not in pooled]
    return pool, excluded


def review_pool_payload(snapshot: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The body of ``GET /api/review-pool`` (contract §1.4) for one settings snapshot; ``row_costs`` prices
    every catalog row (an unmarked row shows its price too); ``migration`` is the lanes-to-pool receipt deciding
    THIS document: ``{snapshot, reported, trigger, outcome: converted|factory|error, error, source: document|
    environment|error|history}``; ``pool_without_credentials``: pool rows whose model has no credentials here."""
    from ouroboros import config as _config
    from ouroboros.configured_subagents import MAX_CONFIGURED_SUBAGENTS, SUBAGENTS_SETTING, parse_configured_subagents, roster_handles
    from ouroboros.provider_models import model_has_credentials_in_settings
    from ouroboros.review_pool_receipts import document_as_read
    from ouroboros.reviewer_slot_config import review_pool_slots, review_pool_state, reviewer_slot_last_executions
    from ouroboros.server_maintenance import review_pool_migration_payload
    from ouroboros.settings_integrity import runtime_environ

    env = dict(runtime_environ() if snapshot is None else snapshot)
    # The receipt is judged against the settings DOCUMENT, not the process projection the pool
    # runs from: the projection carries only the live settings keys, so a lane value a refused
    # migration kept in the document is absent from it and the current receipt would read as
    # history (VD3-06). The retained lanes never return to the runtime configuration here.
    settings_document = env if snapshot is not None else document_as_read(_config.SETTINGS_PATH, env)
    raw = str(env.get(SUBAGENTS_SETTING) or "")
    document = _catalog_document(raw)
    items = document["items"]
    payload: Dict[str, Any] = {
        "limits": {"rows": MAX_CONFIGURED_SUBAGENTS},
        "catalog": {"present": bool(raw.strip()), "enabled": bool(raw.strip()) and document.get("enabled") is not False,
                    "rows": len(items), "eligible": sum(1 for item in items if item.get("review_eligible") is True)},
        "pool": [], "excluded": [], "last_executions": reviewer_slot_last_executions(), "row_costs": {},
        "config_error": "", "migration": review_pool_migration_payload(env, document=settings_document),
        "pool_without_credentials": [],
    }
    state = review_pool_state(raw)
    try:
        if state.get("state") == "error":
            raise ValueError(str(state.get("error") or "the review pool could not be read"))
        slots = review_pool_slots(env)
        handles = roster_handles(parse_configured_subagents(raw), env) if items else {}
    except ValueError as exc:
        payload["config_error"] = str(exc)
        return payload
    payload["row_costs"] = _review_pool_costs(items, env)
    payload["pool"], payload["excluded"] = review_pool_rows(
        items, slots, handles, payload["last_executions"], payload["row_costs"])
    payload["pool_without_credentials"] = [row["subagent_id"] for row in payload["pool"]  # seats log in themselves
                                           if row["cost"]["basis"] != "subscription_seat"
                                           and not model_has_credentials_in_settings(row["route"]["target_id"], env)]
    return payload


@owner_write_guard
async def api_owner_context_mode(request: Request) -> JSONResponse:
    """Persist the owner-selected working context mode.

    Ordinary modes use this owner path; Cyber can also author a generic save.
    The choice applies to subsequent tasks, so no restart is required.
    """
    body = await _json_body_or_empty(request)
    # Off the event loop, under the document lock (held inside): a slow
    # generic save must not be able to freeze the loop THROUGH this
    # endpoint's synchronous lock acquisition.
    return await _run_settings_writer(_api_owner_context_mode_sync, request, body)


def _api_owner_context_mode_sync(request: Request, body: Any) -> JSONResponse:
    from ouroboros import config as _config
    from ouroboros.runtime_mode_policy import runtime_mode_at_least

    raw_mode = str((body or {}).get("mode") or "").strip().lower()
    from ouroboros.context_mode_compat import VALID_CONTEXT_MODES

    if raw_mode not in set(VALID_CONTEXT_MODES):
        return unsaved_error("'mode' must be one of: " + ", ".join(VALID_CONTEXT_MODES), 400)
    next_mode = _config.normalize_context_mode(raw_mode)
    digest = settings_document_digest()
    previous_mode = _config.get_owner_context_mode()
    cyber = runtime_mode_at_least(_config.get_runtime_mode(), "cyber_pro")
    if not cyber and VALID_CONTEXT_MODES.index(next_mode) < VALID_CONTEXT_MODES.index(previous_mode) and _has_running_agent_tasks():
        return unsaved_error(
            "Context mode can only be lowered while Ouroboros is idle. "
            "Wait until no queued or running work remains, then choose the working context.",
            409,
        )

    def _set_context_mode(current: Dict[str, Any]) -> Dict[str, Any]:
        current["OUROBOROS_CONTEXT_MODE"] = next_mode
        # The retired marker survives one compatibility window only as explicit false
        # provenance, so a stored Low carries owner intent while a bare forwarded env
        # Low remains owner Max for the owner's own working window.
        current["OUROBOROS_CONTEXT_MODE_AUTO_LOW"] = "false"
        return current

    with settings_document_mutation():
        # Re-prove the ordinary-mode idle policy under the lock: either the queue
        # or the previous mode may have changed while this writer waited. The
        # digest alone cannot attest idleness. Cyber can choose the next mode
        # during work; existing task snapshots retain their original settings.
        previous_mode = _config.get_owner_context_mode()
        if not cyber and VALID_CONTEXT_MODES.index(next_mode) < VALID_CONTEXT_MODES.index(previous_mode) and _has_running_agent_tasks():
            return unsaved_error(
                "Context mode can only be lowered while Ouroboros is idle. "
                "Wait until no queued or running work remains, then choose the working context.",
                409,
            )
        # This endpoint IS the author of both keys, so they persist even at the shipped default.
        # The digest binds the idle refusal above to the document this write replaces.
        _owner_update_settings(_set_context_mode, digest,
                               authored_keys=_CONTEXT_MODE_KEYS, allow_context_lowering=True)
        # Same-lock projection: see api_owner_auto_grant.
        os.environ["OUROBOROS_CONTEXT_MODE"] = next_mode
        os.environ["OUROBOROS_CONTEXT_MODE_AUTO_LOW"] = "false"
    _owner_audit(
        request,
        "context_mode",
        {"context_mode": next_mode, "previous_context_mode": previous_mode},
    )
    return JSONResponse({"ok": True, "context_mode": next_mode})


@owner_write_guard
async def api_owner_safety_mode(request: Request) -> JSONResponse:
    """Persist the owner-selected LLM-safety-supervisor coverage (full | light | off).

    This dedicated owner path is audited. Ordinary modes skip this control in
    generic settings saves; Cyber also has audited configuration authority there.
    Subsequent operations use the new choice; current task snapshots stay intact."""
    body = await _json_body_or_empty(request)
    # Off the event loop, under the document lock (held inside): a slow
    # generic save must not be able to freeze the loop THROUGH this
    # endpoint's synchronous lock acquisition.
    return await _run_settings_writer(_api_owner_safety_mode_sync, request, body)


def _api_owner_safety_mode_sync(request: Request, body: Any) -> JSONResponse:
    from ouroboros import config as _config

    raw_mode = str((body or {}).get("mode") or "").strip().lower()
    if raw_mode not in set(_config.VALID_SAFETY_MODES):
        return unsaved_error("'mode' must be one of: full, light, off", 400)

    def _set_safety_mode(current: Dict[str, Any]) -> Dict[str, Any]:
        current["OUROBOROS_SAFETY_MODE"] = raw_mode
        return current

    with settings_document_mutation():
        # The previous mode is read for the audit trail only; the digest binds that
        # read to the document the locked update replaces.
        digest = settings_document_digest()
        previous = _config.normalize_safety_mode(
            _owner_read_settings_raw().get("OUROBOROS_SAFETY_MODE"))
        _owner_update_settings(_set_safety_mode, digest,
                               authored_keys=("OUROBOROS_SAFETY_MODE",), allow_safety_lowering=True)
        # Same-lock projection: see api_owner_auto_grant.
        os.environ["OUROBOROS_SAFETY_MODE"] = raw_mode
    _owner_audit(
        request,
        "safety_mode",
        {"safety_mode": raw_mode, "previous_safety_mode": previous},
    )
    return JSONResponse({"ok": True, "safety_mode": raw_mode})


async def api_acknowledge_capability(request: Request) -> JSONResponse:
    """Acknowledge a context window or response maximum for an exact route.
    Provider, model, endpoint and applicable headers/options bind the assertion;
    headless acknowledgements need the same fingerprint, never repo-wide trust.

    Context and output acknowledgements have independent records and timestamps.
    They write the evidence store, never settings.json, so this endpoint deliberately
    has no ``owner_write_guard`` or settings lock and cannot answer ``settings_locked``."""
    body = await _json_body_or_empty(request)
    provider = str((body or {}).get("provider") or "").strip()
    model = str((body or {}).get("model") or "").strip()
    if not provider or not model:
        return json_error("'provider' and 'model' are required", 400)
    if "max_output_tokens" in (body or {}):
        from ouroboros.response_limits import record_response_ack
        try:
            record = await asyncio.to_thread(record_response_ack, request_drive_root(request),
                provider=provider, model=model, base_url=str(body.get("base_url") or ""),
                options=body.get("options"), max_output_tokens=body["max_output_tokens"],
                expected_route_fp=str(body.get("route_fp") or ""))
            _owner_audit(request, "response_cap_ack", record)
            return JSONResponse({"ok": True, "ack": record})
        except (ValueError, TypeError) as exc:
            return json_error(str(exc), 400)
    try:
        window_tokens = int((body or {}).get("window_tokens") or 0)
    except (TypeError, ValueError):
        window_tokens = 0
    if window_tokens <= 0:
        return json_error("'window_tokens' must be a positive integer", 400)
    try:
        from ouroboros.capability_evidence import record_owner_ack
        record = await asyncio.to_thread(record_owner_ack,
            request_drive_root(request),
            provider=provider, model=model,
            base_url=str((body or {}).get("base_url") or ""),
            window_tokens=window_tokens,
            headers=(body or {}).get("headers") if isinstance((body or {}).get("headers"), dict) else None,
            options=(body or {}).get("options") if isinstance((body or {}).get("options"), dict) else None,
            note=str((body or {}).get("note") or ""),
            expected_route_fp=str((body or {}).get("route_fp") or ""),
        )
        _owner_audit(request, "capability_ack", {"route_fp": record.get("route_fp"), "window_tokens": window_tokens, "model": model})
        return JSONResponse({"ok": True, "ack": record})
    except ValueError as exc:
        return json_error(str(exc), 400)
    except Exception as exc:
        return json_exception(exc)


async def api_review_pool(request: Request) -> JSONResponse:
    """GET /api/review-pool — the review pool as Settings → Agents shows it.

    The pool is the enabled catalog rows marked Reviewer, in catalog order, each
    with its effective effort and delivery, its per-review price, and what it
    REALLY ran as last time. A catalog that cannot be read comes back as a typed
    ``config_error`` with an empty pool instead of a 500: the page renders the
    error beside the editor that can fix it. The price estimate may refresh a
    route tariff over the network, so the read runs off the event loop.
    """
    try:
        return JSONResponse(await asyncio.to_thread(review_pool_payload))
    except Exception as exc:
        return json_exception(exc)


async def api_settings_get(request: Request) -> JSONResponse:
    settings, _, _ = apply_runtime_provider_defaults(load_settings())
    if request.query_params.get("websearch_preview") == "1":
        from ouroboros.search_routes import resolve_web_search_route
        return JSONResponse(resolve_web_search_route(settings=settings,
            backend=request.query_params.get("backend"), model=request.query_params.get("model")))
    if request.query_params.get("response_limit_preview") == "1":
        from ouroboros.response_limits import response_limit_preview
        model = str(request.query_params.get("model") or settings.get("OUROBOROS_MODEL") or "")
        route = _active_main_route(settings, model_override=model,
            use_local_override=request.query_params.get("local") == "true")
        return JSONResponse(await asyncio.to_thread(response_limit_preview, request_drive_root(request), route,
            account=request.query_params.get("account", ""), settings=settings))
    safe = {k: v for k, v in settings.items()}
    for key in SECRET_SETTING_KEYS:
        if safe.get(key):
            safe[key] = mask_settings_secret(key, safe[key])
    safe["MCP_SERVERS"] = _mask_mcp_servers_payload(safe.get("MCP_SERVERS") or [])
    for key, value in list(safe.items()):
        if key in SECRET_SETTING_KEYS or key in _SETTINGS_DEFAULTS:
            continue
        if is_custom_secret_setting_key(
            key, known_setting_keys=_SETTINGS_DEFAULTS
        ) and value:
            safe[key] = mask_settings_secret(key, value)
    try:
        port = int(_port_file(request).read_text().strip()) if _port_file(request).exists() else _default_port(request)
    except (ValueError, OSError):
        port = _default_port(request)
    meta = _build_network_meta(_current_bind_host(request), port)
    # Keep the three owner-facing policy axes honest after reload.  The values
    # on the document are the pending/configured choices; process state is the
    # effective value this server can currently report.  Runtime access is
    # restart-bound, while Supervisor and Review are picked up for new tasks by
    # the existing settings effect path.  This is presentation metadata only,
    # not a second policy store.
    try:
        meta["policy_state"] = _build_policy_state(settings)
        meta["restart_state"] = _build_restart_state(settings)
    except Exception:
        # A settings read must stay available even if an optional projection
        # helper is unavailable during startup.  The persisted values remain
        # the ordinary response fields and are still masked below.
        log.debug("Could not build settings policy-state projection", exc_info=True)
    meta["custom_secret_keys"] = sorted(
        key for key in settings
        if key not in SECRET_SETTING_KEYS
        and key not in _SETTINGS_DEFAULTS
        and is_custom_secret_setting_key(key, known_setting_keys=_SETTINGS_DEFAULTS)
        and settings.get(key)
    )
    meta["setup_contract"] = build_setup_contract("web")
    from ouroboros.configured_subagents import (
        configured_subagents_dict,
        resolve_settings_subagent_candidate,
    )
    # Pure API/local defaults only; the editor enriches a still-clean draft
    # with connected sessions through the read-only preview endpoint.
    subagents, candidate_diagnostics = resolve_settings_subagent_candidate(settings)
    meta["available_subagents"] = {
        "source": subagents.source,
        "diagnostic": subagents.diagnostic,
        "diagnostics": candidate_diagnostics,
        "candidate": (
            configured_subagents_dict(subagents.config) if subagents.config is not None else None
        ),
    }
    safe["_meta"] = meta
    return JSONResponse(safe)


async def api_onboarding(request: Request) -> Response:
    """The blocking first-run overlay — a pure READ (D-8).

    Normalization still runs, but only to shape what the wizard DISPLAYS. It is
    deliberately not persisted here: a GET must never be the first author of
    settings.json. Doing so created the file before the owner had answered
    anything, which (a) silently disqualified the fresh-install latch the
    install-time preset and the ``light`` safety default both depend on, and
    (b) made a page load the author of provider defaults the owner never saw.
    The save paths (POST /api/settings, POST /api/onboarding/complete, the
    desktop wizard bridge) keep the same normalization and persist it."""
    from ouroboros.config import SETTINGS_PATH

    settings, _changed, _keys = apply_runtime_provider_defaults(load_settings())
    if has_startup_ready_provider(settings):
        return Response(status_code=204)
    return HTMLResponse(build_onboarding_html(settings, host_mode="web", fresh_install=not SETTINGS_PATH.exists()))


def _apply_settings_save_side_effects(
    request: Request,
    current: Dict[str, Any],
    old_effective_settings: Dict[str, Any],
    all_changed: list,
) -> list:
    """Post-save hot-reload side effects (MCP, extensions, supervisor budgets/timeouts).

    Returns warning strings for side effects that FAILED: an immediate-classed
    key whose hot apply broke must not let the save report "took effect
    immediately" without saying so (#285).
    """
    side_effect_warnings: list = []
    if any(k in all_changed for k in ("MCP_ENABLED", "MCP_SERVERS", "MCP_TOOL_TIMEOUT_SEC")):
        try:
            from ouroboros.mcp_client import (
                reconfigure_from_settings as _mcp_reconfigure,
                refresh_all_background as _mcp_refresh_background,
            )
            _mcp_reconfigure(current)
            _mcp_refresh_background(reason="settings")
        except Exception as exc:
            log.warning("MCP reconfigure after settings change failed", exc_info=True)
            side_effect_warnings.append(
                "MCP reconfigure failed in the server process: "
                f"{type(exc).__name__}: {exc}. The saved values are on disk; "
                "agent processes retry on their next tool-schema read."
            )

    # Skills repo/runtime changes require extension loader reconciliation.
    try:
        from ouroboros.extension_loader import reload_all as _reload_extensions
        new_path = str(current.get("OUROBOROS_SKILLS_REPO_PATH") or "").strip()
        old_path = str(old_effective_settings.get("OUROBOROS_SKILLS_REPO_PATH") or "").strip()
        new_runtime_mode = str(current.get("OUROBOROS_RUNTIME_MODE") or "").strip()
        old_runtime_mode = str(old_effective_settings.get("OUROBOROS_RUNTIME_MODE") or "").strip()
        if new_path != old_path or new_runtime_mode != old_runtime_mode:
            # Use load_settings so extensions do not capture a stale snapshot.
            from ouroboros.config import load_settings as _load_settings
            reload_drive_root = pathlib.Path(
                request.app.state.drive_root
                if hasattr(request.app, "state") and hasattr(request.app.state, "drive_root")
                else request_drive_root(request)
            )
            if (
                (bool(os.environ.get("PYTEST_CURRENT_TEST")) or "pytest" in sys.modules)
                and reload_drive_root == pathlib.Path.home() / "Ouroboros" / "data"
                and not os.environ.get("OUROBOROS_DATA_DIR")
            ):
                log.info("Skipping extension reload_all against real DATA_DIR during pytest settings save")
            else:
                _reload_extensions(
                    reload_drive_root,
                    _load_settings,
                    repo_path=new_path or None,
                )
    except Exception as exc:
        log.error("Extension reload after settings change failed", exc_info=True)
        side_effect_warnings.append(
            "Skills repo reload failed in the server process: "
            f"{type(exc).__name__}: {exc}. The saved path is on disk and "
            "applies after a restart."
        )

    try:
        from supervisor.state import refresh_budget_from_settings
        refresh_budget_from_settings(current)
    except Exception:
        pass
    try:
        from supervisor.queue import refresh_timeouts_from_settings
        refresh_timeouts_from_settings(current)
    except Exception:
        pass
    try:
        from supervisor.message_bus import refresh_budget_limit
        raw_budget = current.get("TOTAL_BUDGET")
        new_budget = float(raw_budget) if raw_budget is not None else 0.0
        refresh_budget_limit(new_budget)
    except Exception:
        pass
    return side_effect_warnings


async def api_settings_post(request: Request) -> JSONResponse:
    # Body parse stays on the loop (it is the only await); everything else runs
    # in a worker thread. The save body is synchronous work — validation, the
    # disk write, env projection, hot-reload side effects, and (when review
    # keys changed) NETWORK evidence fetches for the warning surface — and on
    # the event loop it froze every other request and WebSocket for the whole
    # save, which read as the entire app hanging on the Save button.
    try:
        body = await request.json()
    except Exception as exc:
        # Same answer the broad in-body handler used to give a parse failure.
        return unsaved_error(str(exc), 400)
    return await _run_settings_writer(_api_settings_post_sync, request, body)


def _api_settings_post_sync(request: Request, body: Any) -> JSONResponse:
    # The event loop used to serialize every settings writer for free (no
    # writer awaited mid read-merge-write); a worker thread does not inherit
    # that, so the whole body holds the seam-wide document lock — the
    # single-decision endpoints hold the same lock, and the loop itself stays
    # free to serve everything else.
    with settings_document_mutation():
        return _api_settings_post_locked(request, body)


def _network_settings_error(request: Request, current: dict, old_settings: dict) -> JSONResponse | None:
    """Validate the existing save-time bind/password contract before persistence."""
    try:
        from ouroboros.server_auth import is_loopback_host
        from ouroboros.config import get_runtime_mode
        from ouroboros.runtime_mode_policy import runtime_mode_at_least

        desired_host = str(current.get("OUROBOROS_SERVER_HOST") or "").strip()
        desired_password = str(current.get("OUROBOROS_NETWORK_PASSWORD") or "").strip()
        trust_unauth = (_trust_nonlocal_bind_without_password_enabled()
                        or runtime_mode_at_least(get_runtime_mode(), "cyber_pro"))
        allowed_saved_hosts = {"", "127.0.0.1", "localhost", "::1", "[::1]", "0.0.0.0", "::", "[::]"}
        if desired_host and desired_host not in allowed_saved_hosts:
            return unsaved_error(
                "Server Bind Host in Settings supports localhost or wildcard "
                "binds only (127.0.0.1 or 0.0.0.0). Specific LAN IP binds "
                "are manual/env-only so the desktop launcher can keep using "
                "a reliable loopback health check.",
                400,
            )
        if desired_host and not is_loopback_host(desired_host) and not desired_password and not trust_unauth:
            return unsaved_error(
                "Setting a non-localhost Server Bind Host through the web UI "
                "requires a Network Password in the same save. For manual "
                "trusted-lab/Docker setups, stop Ouroboros and edit "
                "settings.json or environment variables directly.",
                400,
            )
        current_effective_host = (
            str(_current_bind_host(request) or "").strip()
            or str(os.environ.get("OUROBOROS_SERVER_HOST") or "").strip()
        )
        old_password = str(old_settings.get("OUROBOROS_NETWORK_PASSWORD") or "").strip()
        if (
            current_effective_host
            and not is_loopback_host(current_effective_host)
            and old_password
            and not desired_password
            and not trust_unauth
        ):
            return unsaved_error(
                "Cannot clear Network Password while the running server is "
                "still bound to a non-localhost interface. First save a "
                "loopback Server Bind Host and restart, then clear the password.",
                400,
            )
    except Exception:
        log.warning("Could not validate network bind settings", exc_info=True)
    return None


def _effort_range_tiers(body: Dict[str, Any]) -> tuple[Dict[str, Any], str]:
    """Lower-case each supplied effort-range key (benchmarks and Cyber write TASK here) and
    name the first non-tier; order is not judged on this path, the read is tolerant
    (`POST /api/owner/effort-range` is the strict writer)."""
    from ouroboros.settings_scales import EFFORT_RANGE_KEYS, EFFORT_SCALE

    for effort_key in (key for key in EFFORT_RANGE_KEYS if key in body):
        tier = str(body.get(effort_key) or "").strip().lower()
        if tier not in EFFORT_SCALE:
            return body, f"{effort_key} must be one of: {', '.join(EFFORT_SCALE)}."
        body = dict(body)
        body[effort_key] = tier
    return body, ""


def _api_settings_post_locked(request: Request, body: Any) -> JSONResponse:
    # Everything below the write is a POST-commit step. The broad handler at the
    # bottom used to answer a failure there with "400, nothing saved" while the
    # bytes were already on disk; `boundary` is what lets it tell the two apart.
    boundary = CommitBoundary()
    try:
        if not isinstance(body, dict):
            return unsaved_error("JSON body must be an object.", 400)
        body = dict(body)  # the owner's empty-pool confirmation is a request flag, never a setting
        allow_empty_pool = body.pop(ALLOW_EMPTY_REVIEW_POOL, None) is True
        channel_key = "OUROBOROS_UPDATE_CHANNEL"
        if channel_key in body:
            from ouroboros.update_channels import UPDATE_CHANNEL_BRANCHES

            raw_channel = str(body.get(channel_key) or "").strip().lower()
            if raw_channel not in UPDATE_CHANNEL_BRANCHES:
                return unsaved_error(
                    f"{channel_key} must be one of: stable, qa, development.", 400
                )
            body = dict(body)
            body[channel_key] = raw_channel
        # Reject a malformed post-task evolution cadence at the API boundary: the
        # read-time getter only normalizes, and the Settings UI validates its own Save,
        # but a direct API client must not be able to persist e.g. every_n:0 or garbage.
        cadence_key = "OUROBOROS_POST_TASK_EVOLUTION_CADENCE"
        if cadence_key in body:
            from ouroboros import config as _config
            raw_cadence = str(body.get(cadence_key) or "").strip()
            if raw_cadence and not _config.is_valid_post_task_evolution_cadence(raw_cadence):
                return unsaved_error(f"{cadence_key} must be one of: off, llm, every_n:<positive int>.", 400)
        # Shared review-cycle cap: same boundary rule as the cadence — the read-time
        # getter fails closed to the default, the UI offers only valid segments, but a
        # direct API client must not persist garbage. Aliases canonicalize to "unlimited".
        from ouroboros.review_cycles import (
            REVIEW_MAX_CYCLES_KEY, is_valid_review_max_cycles, normalize_review_max_cycles,
        )
        if REVIEW_MAX_CYCLES_KEY in body:
            raw_cycles = str(body.get(REVIEW_MAX_CYCLES_KEY) or "").strip()
            if raw_cycles and not is_valid_review_max_cycles(raw_cycles):
                return unsaved_error(
                    f"{REVIEW_MAX_CYCLES_KEY} must be a positive integer or 'unlimited'.", 400
                )
            if raw_cycles:
                body = dict(body)
                body[REVIEW_MAX_CYCLES_KEY] = normalize_review_max_cycles(raw_cycles)
        # Optional task bounds (round limit, absolute lifetime): the same vocabulary, but a
        # blank, zero or malformed value is refused rather than read as "no limit" or as a
        # silent default; a valid value persists as an int or the canonical "unlimited".
        from ouroboros.settings_scales import OPTIONAL_BOUND_LEGACY, UNLIMITED, parse_positive_or_unlimited
        for bound_key in (key for key in OPTIONAL_BOUND_LEGACY if key in body):
            try:
                bound = parse_positive_or_unlimited(body.get(bound_key))
            except (TypeError, ValueError):
                return unsaved_error(f"{bound_key} must be a positive integer or 'unlimited'.", 400)
            body = dict(body)
            body[bound_key] = UNLIMITED if bound is None else bound
        body, effort_error = _effort_range_tiers(body)
        if effort_error:
            return unsaved_error(effort_error, 400)
        # The catalog is judged as THIS save produces it: twins, then the review pool.
        subagents_key = "OUROBOROS_SUBAGENTS"
        catalog_saved = subagents_key in body and body.get(subagents_key) not in (None, "")
        if catalog_saved:
            from ouroboros.configured_subagents import (
                normalize_configured_subagents, roster_save_error,
            )
            try:
                _subagents, canonical_subagents = normalize_configured_subagents(
                    body.get(subagents_key)
                )
            except ValueError as exc:
                return unsaved_error(str(exc), 400)
            stored_for_rules = load_settings()
            # Twins are refused only when THIS save changes the roster.
            twin_error = roster_save_error(canonical_subagents, stored_for_rules, body)
            if twin_error:
                return unsaved_error(twin_error, 400)
            pool_error = review_pool_save_judgement(canonical_subagents, stored_for_rules, allow_empty=allow_empty_pool)
            if pool_error:
                return unsaved_error(pool_error, 400, code="empty_review_pool")
            body = dict(body)
            body[subagents_key] = canonical_subagents
        parsed_budget: dict[str, float] = {}
        for budget_key in BUDGET_SETTING_KEYS:
            if budget_key not in body:
                continue
            budget_value, budget_error = parse_budget_setting(budget_key, body.get(budget_key))
            if budget_error:
                return unsaved_error(budget_error, 400)
            if budget_value is not None:
                parsed_budget[budget_key] = budget_value
        if parsed_budget:
            body = dict(body)
            body.update(parsed_budget)
        old_settings = load_settings()
        from ouroboros.config import get_runtime_mode, normalize_runtime_mode as _norm_runtime_mode

        raw_old_settings = _owner_read_settings_raw()
        pending_runtime_mode = _norm_runtime_mode(
            raw_old_settings.get("OUROBOROS_RUNTIME_MODE", old_settings.get("OUROBOROS_RUNTIME_MODE"))
        )
        current_runtime_mode = get_runtime_mode()
        old_effective_settings = dict(old_settings)
        old_effective_settings["OUROBOROS_RUNTIME_MODE"] = current_runtime_mode
        if "MCP_SERVERS" in body:
            body = dict(body)
            try:
                body["MCP_SERVERS"] = _rehydrate_mcp_servers_payload(
                    body.get("MCP_SERVERS"),
                    old_settings.get("MCP_SERVERS"),
                )
            except MCPSecretIdentityAmbiguous as exc:
                return unsaved_error(str(exc), 409, code=exc.code)
        current = _merge_settings_payload(old_effective_settings, body)
        from ouroboros.runtime_mode_policy import runtime_mode_at_least

        requested_runtime_mode = _norm_runtime_mode(current.get("OUROBOROS_RUNTIME_MODE"))
        runtime_authored = "OUROBOROS_RUNTIME_MODE" in body and runtime_mode_at_least(current_runtime_mode, "cyber_pro")
        runtime_changed = runtime_authored and requested_runtime_mode != pending_runtime_mode
        if runtime_authored:
            pending_runtime_mode = requested_runtime_mode
        minimax_region = str(current.get("MINIMAX_REGION") or "").strip().lower()
        if minimax_region and minimax_region not in MINIMAX_REGION_ENDPOINTS:
            return unsaved_error("MINIMAX_REGION must be global_en or cn_zh.", 400)
        current["MINIMAX_REGION"] = minimax_region
        zai_plan = str(current.get("ZAI_PLAN") or "").strip().lower()
        if zai_plan and zai_plan not in ZAI_PLAN_ENDPOINTS:
            return unsaved_error("ZAI_PLAN must be payg or coding.", 400)
        current["ZAI_PLAN"] = zai_plan
        # Generic settings saves operate on the current boot baseline. A pending
        # next-boot mode written by /api/owner/runtime-mode is preserved on disk
        # below, but never hot-applied to this process/env.
        current["OUROBOROS_RUNTIME_MODE"] = current_runtime_mode
        # Trim opaque path text so configured/empty state is deterministic.
        current["OUROBOROS_SKILLS_REPO_PATH"] = str(
            current.get("OUROBOROS_SKILLS_REPO_PATH") or ""
        ).strip()
        network_error = _network_settings_error(request, current, old_settings)
        if network_error is not None:
            return network_error
        current, provider_defaults_changed, provider_default_keys = apply_runtime_provider_defaults(current)
        if str(current.get("LOCAL_MODEL_SOURCE", "") or "").strip() and not has_startup_ready_provider(current):
            return unsaved_error("Local-only setups must route at least one model to the local runtime.", 400)
        # A catalog save retires the former review lanes: the pool replaces them.
        if catalog_saved:
            current.pop(REVIEW_LANES_KEY, None)
        all_changed = [
            k for k in current
            if str(current.get(k, "") or "") != str(old_effective_settings.get(k, "") or "")
        ]
        if catalog_saved and str(old_effective_settings.get(REVIEW_LANES_KEY) or "").strip():
            all_changed.append(REVIEW_LANES_KEY)
        if runtime_changed:
            all_changed.append("OUROBOROS_RUNTIME_MODE")

        # Snapshot BEFORE the save lands: only a task already started at that
        # moment keeps the previous configuration. Measuring after the write
        # would misreport a task that started in between (it re-reads the NEW
        # settings in handle_task) as one that kept the old. Disclosed residual
        # (adjudicated 2026-08-05): the opposite ms-interleaving exists too — a
        # task that reads settings idle-side just before this save and flips
        # busy just after it gets no warning; a silent miss in that window
        # beats a false "keeps the old config" over a task that has the new.
        # Linearizing properly would need a settings-generation handshake —
        # machinery a warning string does not justify.
        started_before_save = _has_started_agent_tasks()
        settings_to_save = dict(current)
        settings_to_save["OUROBOROS_RUNTIME_MODE"] = pending_runtime_mode
        # Only an explicit Cyber context choice authors the pair; an unrelated save
        # preserves omission and cannot change a running task's captured settings.
        authored_keys = tuple(key for key in ("OUROBOROS_SAFETY_MODE",) if key in all_changed)
        if runtime_mode_at_least(current_runtime_mode, "cyber_pro") and "OUROBOROS_CONTEXT_MODE" in body:
            authored_keys += tuple(_CONTEXT_MODE_KEYS)
        _owner_write_settings(
            settings_to_save,
            authored_keys=authored_keys,
            boundary=boundary)
        control_changes = {key: {"old": raw_old_settings.get(key), "new": settings_to_save.get(key)}
                           for key in ("OUROBOROS_RUNTIME_MODE", "OUROBOROS_SAFETY_MODE",
                                       "OUROBOROS_AUTO_GRANT_REVIEWED_SKILLS",
                                       "OUROBOROS_CONTEXT_MODE", "OUROBOROS_REVIEW_ENFORCEMENT")
                           if key in all_changed}
        if control_changes:
            _owner_audit(request, "settings_controls", {"changes": control_changes})
        boundary.at("environment projection")
        _apply_settings_to_env(current)
        boundary.at("supervisor start")
        _start_supervisor_if_needed_for_request(request, current)

        boundary.at("hot-reload")
        side_effect_warnings = _apply_settings_save_side_effects(
            request, current, old_effective_settings, all_changed)
        boundary.at("post-save notices")

        # Tolerate stubbed side effects returning None (test harnesses).
        warnings = list(side_effect_warnings or [])
        if provider_defaults_changed:
            change_kind = classify_runtime_provider_change(old_effective_settings, current)
            if change_kind == "direct_normalize":
                warnings.append(
                    "Normalized direct-provider routing because OpenRouter is not configured for the active provider."
                )
        try:
            from supervisor.message_bus import get_bridge
            get_bridge().configure_from_settings(current)
        except Exception:
            pass
        try:
            from ouroboros.server_auth import is_loopback_host
            desired_host = str(current.get("OUROBOROS_SERVER_HOST") or "").strip()
            desired_password = str(current.get("OUROBOROS_NETWORK_PASSWORD") or "").strip()
            if desired_host and not is_loopback_host(desired_host) and not desired_password:
                if _trust_nonlocal_bind_without_password_enabled():
                    warnings.append(
                        "OUROBOROS_TRUST_NONLOCAL_BIND_WITHOUT_PASSWORD=1 allows this "
                        "non-localhost bind without Ouroboros's internal Network Password. "
                        "Use only behind ingress auth, VPN, private networking, or an auth proxy."
                    )
                else:
                    warnings.append(
                        "Server Bind Host is non-localhost and Network Password is empty; "
                        "after restart the app will be reachable on the network without a password."
                    )
        except Exception:
            pass
        _repo_slug = current.get("GITHUB_REPO", "")
        _gh_token = current.get("GITHUB_TOKEN", "")
        if _gh_token and any(k in all_changed for k in ("GITHUB_REPO", "GITHUB_TOKEN")):
            boundary.at("GitHub remote configuration")
            from supervisor.git_ops import configure_personal_remote
            remote_ok, remote_msg, resolved_slug = configure_personal_remote(
                _repo_slug,
                _gh_token,
                auto_fork=not bool(str(_repo_slug or "").strip()),
                confirm_replace_origin=bool(body.get("GITHUB_REPLACE_ORIGIN_CONFIRMED")),
            )
            if not remote_ok:
                log.warning("Remote configuration failed on settings save: %s", remote_msg)
                warnings.append(f"Remote config failed: {remote_msg}")
            elif resolved_slug and resolved_slug != _repo_slug:
                current["GITHUB_REPO"] = resolved_slug
                settings_to_save["GITHUB_REPO"] = resolved_slug
                _owner_write_settings(settings_to_save)
                os.environ["GITHUB_REPO"] = resolved_slug
        immediate_changed, next_task_changed = _effect_buckets(all_changed)
        agent_task_running = bool(next_task_changed) and started_before_save
        if agent_task_running:
            # Owner decision (2026-08-05): the task-start snapshot boundary STAYS —
            # a running task keeps the config it started with, and the save says so
            # loudly instead of letting "Saved" read as "applied to the task you are
            # watching" (the reviewer-slot save at 21:56 read exactly that way).
            warnings.append(
                "An agent task is running right now: it keeps the configuration it "
                "started with (models, reviewers, subagents). The saved changes apply "
                "from the next task."
            )
        resp: Dict[str, Any] = {"status": "saved"}
        # A key with its own writer is skipped here, and the save says so instead of reading as applied.
        ignored = sorted(k for k in body if k in _ENDPOINT_WRITTEN_SETTINGS)
        if ignored:
            resp["ignored_keys"] = {k: _ENDPOINT_WRITERS.get(k, "its own endpoint") for k in ignored}
        if agent_task_running:
            resp["agent_task_running"] = True
        if not all_changed:
            resp["no_changes"] = True
        restart_state = _build_restart_state(settings_to_save)
        resp["restart_state"] = restart_state
        if restart_state["restart_required"]:
            resp["restart_required"] = True
            resp["restart_keys"] = restart_state["restart_keys"]
        if immediate_changed:
            resp["immediate_changed"] = True
        if next_task_changed:
            resp["next_task_changed"] = True
        if subagents_key in all_changed:
            _unknown = _unrecognised_review_models(_candidate_pool_api_models(current))
            if _unknown:
                warnings.append(
                    "Unrecognised review model id(s) the provider catalog does not list: "
                    + ", ".join(sorted(set(_unknown)))
                    + ". Review calls to these reviewers will fail with 'not a valid model ID' "
                    "and can break the review quorum — check for a truncated value."
                )
        from ouroboros.subagent_runtime import review_pool_save_warning
        warnings.extend(filter(None, [review_pool_save_warning(current)]))
        if warnings:
            resp["warnings"] = warnings
        return JSONResponse(resp)
    except Exception as e:
        if boundary.committed:
            # The bytes ARE on disk. Reporting this as a failed save would send
            # the owner looking for changes that landed (BIBLE P1). This branch
            # comes FIRST so a post-commit lock refusal is not misread as one.
            return post_commit_failure_response(e, boundary)
        if isinstance(e, SettingsLockUnavailable):
            return unsaved_error(str(e), 503, code="settings_locked")
        if isinstance(e, SettingsDocumentBusy):
            return unsaved_error(str(e), 503, code="settings_busy")
        return unsaved_error(str(e), 400)

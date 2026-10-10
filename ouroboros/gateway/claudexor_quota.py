"""Explicit account refresh/reset transport through the owned Claudexor daemon.

Discovers the already-running daemon, handshakes under the client's default 60 s
read bound, and negotiates resource operations from the catalog. Quota refresh
uses its 90 s foreground bound, preserves the legacy envelope when unsupported,
and accepts an exact account only when advertised. Reset POST retains its body
and Idempotency-Key; GET inspects the existing receipt. Catalog failures retain
their typed cause rather than claiming unsupported. These routes start no daemon,
retry nothing, apply no vendor policy and never expose the daemon token.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict

from starlette.requests import Request
from starlette.responses import JSONResponse

from ouroboros.gateway._helpers import json_error
from ouroboros.gateways.claudexor import ClaudexorUnavailable, account_resource_capabilities

log = logging.getLogger(__name__)


def _capabilities(gateway) -> dict[str, bool]:
    return account_resource_capabilities(gateway.operations())


def _refresh_quota(target: dict | None = None) -> Dict[str, Any]:
    """Call the owned daemon's foreground quota operation exactly once."""
    from ouroboros.claudexor_daemon import owned_config_dir
    from ouroboros.gateways.claudexor import ClaudexorGateway, discover_daemon_at

    endpoint = discover_daemon_at(owned_config_dir())
    with ClaudexorGateway(endpoint) as gateway:
        gateway.handshake()
        capabilities = _capabilities(gateway)
        if target is not None and not capabilities["refresh"]:
            raise ClaudexorUnavailable("account_resources_unsupported",
                                      "This engine does not support exact-account refresh", status_code=503)
        if capabilities["refresh"]:
            return gateway.refresh_quota(target=target, view="resources")
        return gateway.refresh_quota()


def _account_reset(*, request: dict | None = None, key: str = "", operation_id: str = "") -> dict:
    from ouroboros.claudexor_daemon import owned_config_dir
    from ouroboros.gateways.claudexor import ClaudexorGateway, discover_daemon_at

    with ClaudexorGateway(discover_daemon_at(owned_config_dir())) as gateway:
        gateway.handshake()
        capability = "inspect_reset" if operation_id else "reset"
        if not _capabilities(gateway)[capability]:
            raise ClaudexorUnavailable("account_resets_unsupported",
                                      "This engine does not support account reset operations", status_code=503)
        if operation_id:
            return gateway.get_account_reset(operation_id)
        return gateway.create_account_reset(request, idempotency_key=key)


def _target_valid(target: Any) -> bool:
    return (isinstance(target, dict) and set(target) == {"harness", "profile_id"}
            and all(isinstance(value, str) and value.strip() for value in target.values()))


def _problem(exc: ClaudexorUnavailable) -> JSONResponse:
    """Preserve the engine's typed refusal without exposing private transport state."""
    body = {"error": str(exc), "code": exc.code, "required_actions": list(exc.required_actions)}
    headers = {"Retry-After": exc.retry_after} if exc.retry_after else None
    return JSONResponse(body, status_code=exc.status_code if 400 <= exc.status_code < 600 else 503,
                        headers=headers)


async def api_claudexor_quota_refresh(request: Request) -> JSONResponse:
    """POST /api/claudexor/quota/refresh — explicit owner foreground refresh."""
    try:
        # The established body-less full refresh remains valid.
        raw = await request.body()
        body = json.loads(raw) if raw else {}
        if not isinstance(body, dict) or set(body) - {"target"}:
            return json_error("Expected an optional exact account target", 400)
        if "target" in body:
            if not _target_valid(body["target"]):
                return json_error("target requires non-empty harness and profile_id strings", 400)
            return JSONResponse(await asyncio.to_thread(_refresh_quota, body["target"]))
        return JSONResponse(await asyncio.to_thread(_refresh_quota))
    except (ValueError, TypeError):
        return json_error("Invalid JSON request", 400)
    except ClaudexorUnavailable as exc:
        return _problem(exc)
    except Exception as exc:
        log.exception("api_claudexor_quota_refresh failed")
        return json_error(f"{type(exc).__name__}: Claudexor quota refresh failed")


async def api_claudexor_account_reset(request: Request) -> JSONResponse:
    """Explicit reset or receipt inspection; no account admission or retry policy."""
    try:
        if request.method == "GET":
            return JSONResponse(await asyncio.to_thread(
                _account_reset, operation_id=request.path_params["operation_id"]))
        body = await request.json()
        if (not isinstance(body, dict) or set(body) - {"target", "offer_id", "grant_id"}
                or not _target_valid(body.get("target"))
                or not isinstance(body.get("offer_id"), str) or not body["offer_id"].strip()
                or ("grant_id" in body and (not isinstance(body["grant_id"], str)
                                           or not body["grant_id"].strip()))):
            return json_error("Expected target, offer_id and optional grant_id", 400)
        key = request.headers.get("Idempotency-Key", "")
        if not key.strip():
            return json_error("Idempotency-Key is required", 400)
        return JSONResponse(await asyncio.to_thread(_account_reset, request=body, key=key))
    except (ValueError, TypeError):
        return json_error("Invalid JSON request", 400)
    except ClaudexorUnavailable as exc:
        return _problem(exc)
    except Exception:
        log.exception("api_claudexor_account_reset failed")
        return json_error("Claudexor account reset request failed", 503)


__all__ = ["api_claudexor_quota_refresh", "api_claudexor_account_reset"]

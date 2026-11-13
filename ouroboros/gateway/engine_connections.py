"""``/api/engine-connections``: the owner's view, Check and Disconnect of named remote engines (``remote_engines``).

The list reads saved rows and the installation's last observation only (plus a removed connection Emergency Stop
still owes a stop on); it never contacts a host. Check attaches when no tunnel answers (attach-only: a running
engine's descriptor and token are read, nothing is started) and reuses a live one; Disconnect stops the
installation's forward only, for any name the connection grammar allows (a removed row's tunnel included). The
catalog answers through a tunnel that is already open and never opens one: an engine not connected is ``offline``,
not empty. No answer carries the token.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List

from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from ouroboros.gateway._helpers import json_error


def _name(request: Request) -> str:
    return str(request.path_params.get("name") or "")


async def api_engine_connections(_request: Request) -> JSONResponse:
    from ouroboros import remote_engines

    return JSONResponse({"connections": await asyncio.to_thread(remote_engines.status)})


async def api_engine_connection_check(request: Request) -> JSONResponse:
    from ouroboros import remote_engines

    name = _name(request)
    if not any(row.get("name") == name for row in remote_engines.engine_connection_rows()):
        return json_error(f"no saved engine connection is named {name!r}; save it first", 404)
    return JSONResponse(await asyncio.to_thread(remote_engines.check, name))


async def api_engine_connection_disconnect(request: Request) -> JSONResponse:
    from ouroboros import remote_engines

    name = _name(request)
    if not remote_engines.CONNECTION_NAME.fullmatch(name):  # a removed row's name still closes its tunnel
        return json_error(f"{name!r} is not an engine connection name", 404)
    # ``closed`` is True (stopped), False (none was running) or None (the stop is unconfirmed: never said closed).
    return JSONResponse({"name": name, "closed": await asyncio.to_thread(remote_engines.retire, name)})


def _catalog(name: str) -> Dict[str, Any]:
    """What a connected engine serves: raw model sources with their models (Main, API rows), agent harnesses
    (session rows), each model a route id composed by ``provider_models.claudexor_route``, and that engine's own
    account roster (the pins its routes may name; never this computer's accounts)."""
    from ouroboros import remote_engines
    from ouroboros.gateways.claudexor import ClaudexorUnavailable
    from ouroboros.provider_models import claudexor_route

    try:
        client = remote_engines.gateway(name, new_work=False, attach=False)
    except ClaudexorUnavailable as exc:
        return {"name": name, "state": "offline" if exc.code == "remote_engine_offline" else "unavailable",
                "code": exc.code, "message": str(exc)[:400]}
    sources: List[Dict[str, Any]] = []
    errors: List[Dict[str, str]] = []
    try:
        for source in client.list_model_sources().get("sources") or []:
            identifier = str((source or {}).get("id") or "")
            if not identifier:
                continue
            try:
                models = client.list_source_models(identifier).get("models") or []
            except ClaudexorUnavailable as exc:
                errors.append({"source": identifier, "code": exc.code, "message": str(exc)[:300]})
                models = []
            sources.append({"id": identifier, "label": str(source.get("label") or identifier),
                            "credentialHarness": str(source.get("credentialHarness") or ""), "models": [
                {"id": str(model.get("id")), "label": str(model.get("label") or model.get("id")),
                 "value": claudexor_route(name, identifier, str(model.get("id")))}
                for model in models if isinstance(model, dict) and model.get("id")]})
        harnesses = sorted(str(row.get("id") or "") for row in client.harnesses() if row.get("id"))
        roster = _roster(client)
    except ClaudexorUnavailable as exc:
        return {"name": name, "state": "unavailable", "code": exc.code, "message": str(exc)[:400]}
    finally:
        client.close()
    return {"name": name, "state": "connected", "sources": sources, "harnesses": harnesses, "errors": errors,
            **roster}


def _roster(client: Any) -> Dict[str, Any]:
    """The remote engine's own accounts (``GET /v2/credential-profiles``, judged by the reader the local roster uses):
    ``accounts_state`` ``ok`` with the rows, or ``unread`` with the engine's code, never an empty list for unread."""
    from ouroboros.gateway.claudexor_passive import roster_error
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    try:
        roster = client.credential_profiles()
    except ClaudexorUnavailable as exc:
        return {"accounts_state": "unread", "accounts_code": exc.code}
    if error := roster_error(roster):
        return {"accounts_state": "unread", "accounts_code": error}
    accounts = []
    for wrapper in roster["profiles"]:
        profile, status = wrapper["profile"], wrapper.get("status") if isinstance(wrapper.get("status"), dict) else {}
        accounts.append({"harness_id": profile["harness_id"], "profile_id": profile["profile_id"],
                         "label": str(profile.get("display_name") or profile["profile_id"]),
                         "enabled": profile.get("enabled") is not False,
                         "verification": str(status.get("verification") or "unknown"),
                         "availability": str(status.get("availability") or "unknown")})
    return {"accounts_state": "ok", "accounts": accounts}


async def api_engine_connection_catalog(request: Request) -> JSONResponse:
    return JSONResponse(await asyncio.to_thread(_catalog, _name(request)))


def engine_connection_routes() -> List[Route]:
    return [
        Route("/api/engine-connections", endpoint=api_engine_connections, methods=["GET"]),
        Route("/api/engine-connections/{name}/check", endpoint=api_engine_connection_check, methods=["POST"]),
        Route("/api/engine-connections/{name}/disconnect", endpoint=api_engine_connection_disconnect,
              methods=["POST"]),
        Route("/api/engine-connections/{name}/catalog", endpoint=api_engine_connection_catalog, methods=["GET"]),
    ]


__all__ = ["engine_connection_routes"]

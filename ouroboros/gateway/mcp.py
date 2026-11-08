"""HTTP surface for the shared MCP manager used by Settings and ToolRegistry."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict

from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from ouroboros.config import load_settings
from ouroboros.gateway._helpers import json_error, request_json_or
from ouroboros.mcp_client import (
    canonical_server_id,
    get_manager,
    raw_server_id,
    reconfigure_from_settings,
)
from ouroboros.mcp_headers import MCPHeaderPlaceholderUnmatched

log = logging.getLogger(__name__)


def _ensure_configured() -> None:
    """Reconcile manager with settings; cheap guard against out-of-band edits."""
    try:
        reconfigure_from_settings(load_settings())
    except Exception:
        log.warning("MCP reconfigure_from_settings failed", exc_info=True)


def _status_with_weights() -> Dict[str, Any]:
    """The status snapshot plus each server's catalog potential (``tool_connections.catalog_weights``)."""
    from ouroboros.tool_connections import catalog_weights

    payload = get_manager().status_payload()
    weights = catalog_weights()
    for server in payload["servers"]:
        server.update(weights["servers"].get(server["id"], {}))
    payload["weights"] = weights["totals"]
    return payload


async def api_mcp_status(request: Request) -> JSONResponse:
    """GET /api/mcp/status — masked status snapshot for the UI."""
    try:
        await asyncio.to_thread(_ensure_configured)
        payload = await asyncio.to_thread(_status_with_weights)
        return JSONResponse(payload)
    except Exception as exc:
        log.exception("api_mcp_status failed")
        return json_error(f"{type(exc).__name__}: MCP status failed")


async def api_mcp_refresh(request: Request) -> JSONResponse:
    """POST /api/mcp/refresh — refresh one or all servers."""
    try:
        body: Dict[str, Any] = await request_json_or(request, {})
        if not isinstance(body, dict):
            return json_error("request body must be a JSON object", 400)
        server_id = canonical_server_id(body.get("server_id") or "")
        await asyncio.to_thread(_ensure_configured)
        manager = get_manager()
        if server_id:
            outcome = await asyncio.to_thread(manager.refresh_server, server_id)
            return JSONResponse({"server_id": server_id, **outcome})
        outcome = await asyncio.to_thread(manager.refresh_all)
        return JSONResponse(outcome)
    except Exception as exc:
        log.exception("api_mcp_refresh failed")
        return json_error(f"{type(exc).__name__}: MCP refresh failed")


def _request_base(request: Request) -> str:
    """Where this browser reached the server (its Origin, the deployment prefix kept): the sign-in returns there."""
    origin = request.headers.get("origin") or f"{request.url.scheme}://{request.url.netloc}"
    return origin.rstrip("/") + str(request.scope.get("root_path") or "")


async def api_mcp_sign_in(request: Request) -> JSONResponse:
    """POST /api/mcp/sign-in — one step of a remote server's sign-in (``mcp_oauth``); ``complete`` lists it."""
    from ouroboros import mcp_oauth

    body = await request_json_or(request, None)
    if not isinstance(body, dict) or body.get("action") not in ("begin", "complete", "sign_out"):
        return json_error("supply action begin, complete or sign_out", 400)
    try:
        if body.get("returned_url"):
            body = {**body, **mcp_oauth.parse_return(body["returned_url"])}
        state = str(body.get("state") or "")
        server_id = (mcp_oauth.server_of(state) if body["action"] == "complete"
                     else canonical_server_id(body.get("server_id") or ""))
    except ValueError as exc:
        return json_error(str(exc), 400)
    await asyncio.to_thread(_ensure_configured)
    manager = get_manager()
    cfg = manager.server_config(server_id)
    if cfg is None or not mcp_oauth.applies(cfg):
        return json_error(f"no remote MCP server {server_id!r} that signs in", 404)

    def run(step):  # its own loop on a worker thread: the sign-in's file lock never blocks this one
        return asyncio.to_thread(asyncio.run, step)

    try:
        if body["action"] == "begin":
            result = await run(mcp_oauth.begin(cfg, return_base=_request_base(request),
                                               client_id=str(body.get("client_id") or ""),
                                               client_secret=str(body.get("client_secret") or "")))
        elif body["action"] == "complete":
            result = await run(mcp_oauth.complete(
                cfg, state=state, code=str(body.get("code") or ""), iss=body.get("iss"),
                error=str(body.get("error") or ""), error_description=str(body.get("error_description") or "")))
            listing = await asyncio.to_thread(manager.refresh_server, cfg.id)
            result["listing"] = {key: listing[key] for key in ("ok", "tool_count", "error") if key in listing}
        else:
            result = await run(mcp_oauth.sign_out(cfg))
        return JSONResponse({"ok": True, **result})
    except mcp_oauth.MCPAuthRequired as exc:
        return JSONResponse({"ok": False, "server": cfg.id, "code": exc.code, "reason": exc.reason,
                             "error": exc.detail or exc.reason})
    except Exception as exc:
        log.exception("api_mcp_sign_in failed")
        return json_error(f"{type(exc).__name__}: MCP sign-in failed")


async def api_mcp_test(request: Request) -> JSONResponse:
    """Probe the edited candidate with the same URL rehydration as Settings."""
    from ouroboros.gateway.secret_rows import MCPSecretIdentityAmbiguous, rehydrate_mcp_servers as _rehydrate_mcp_servers_payload

    try:
        body: Dict[str, Any] = await request_json_or(request, {})
        if not isinstance(body, dict):
            return json_error("request body must be a JSON object", 400)
        await asyncio.to_thread(_ensure_configured)
        manager = get_manager()
        server_id = canonical_server_id(body.get("server_id") or "")
        settings = await asyncio.to_thread(load_settings)
        if server_id:
            servers = settings.get("MCP_SERVERS") or []
            # The loader's identity (id, slug or name), so a saved server without an
            # explicit id is found; several claimants never lend one of their secrets.
            matches = [dict(entry) for entry in servers if raw_server_id(entry) == server_id]
            if len(matches) > 1:
                return JSONResponse(
                    {"ok": False, "code": "MCP_ID_AMBIGUOUS",
                     "error": f"{len(matches)} saved servers resolve to server id {server_id!r}"},
                    status_code=409,
                )
            if not matches:
                return JSONResponse(
                    {"ok": False, "error": f"server id {server_id!r} not found"},
                    status_code=404,
                )
            target: Dict[str, Any] = matches[0]
            candidate = body.get("server")
            if isinstance(candidate, dict):
                # Use the edited candidate, but rehydrate masked token
                # values from the saved config. The caller can also omit
                # auth_token entirely to intentionally test without auth.
                probe = _rehydrate_mcp_servers_payload([candidate], [target])[0]
                target = probe
            outcome = await asyncio.to_thread(manager.test_server, target, settings=settings)
            return JSONResponse(outcome)
        candidate = body.get("server")
        if not isinstance(candidate, dict):
            return JSONResponse(
                {"ok": False, "error": "request body must include `server` (object) or `server_id` (string)"},
                status_code=400,
            )
        # Without a selected saved server, URL masks cannot identify credentials.
        candidate = _rehydrate_mcp_servers_payload([candidate], [])[0]
        outcome = await asyncio.to_thread(manager.test_server, candidate, settings=settings)
        return JSONResponse(outcome)
    except (MCPHeaderPlaceholderUnmatched, MCPSecretIdentityAmbiguous) as exc:
        return JSONResponse({"ok": False, "code": exc.code, "error": str(exc)}, status_code=409)
    except Exception as exc:
        log.exception("api_mcp_test failed")
        return json_error(f"{type(exc).__name__}: MCP test failed")


async def api_mcp_import_preview(request: Request) -> JSONResponse:
    """Translate pasted client JSON into draft patches; no settings or transport I/O.

    ``servers`` is the draft's matching metadata (``MCPImportDraftServer``):
    identities, transports and header names, never header values or addresses.
    """
    from ouroboros.gateway.mcp_import_contracts import MCPImportPreviewRequest
    from ouroboros.gateway.schema import validate_ingress
    from ouroboros.mcp_import import preview_import

    headers = {"Cache-Control": "no-store"}
    body = await request_json_or(request, None)
    # Executable gateway ABI: the derived request schema is the ingress gate.
    schema_errors = validate_ingress(body, MCPImportPreviewRequest)
    if schema_errors:
        return JSONResponse({"ok": False, "error": f"Supply text and a servers list: {schema_errors[0]}.",
                             "entries": []}, status_code=400, headers=headers)
    return JSONResponse(preview_import(body["text"], body["servers"]), headers=headers)


def mcp_routes() -> list[Route]:
    """The MCP family: saved status and refresh, a remote server's sign-in, the draft Test and the import preview."""
    return [
        Route("/api/mcp/status", endpoint=api_mcp_status, methods=["GET"]),
        Route("/api/mcp/refresh", endpoint=api_mcp_refresh, methods=["POST"]),
        Route("/api/mcp/sign-in", endpoint=api_mcp_sign_in, methods=["POST"]),
        Route("/api/mcp/test", endpoint=api_mcp_test, methods=["POST"]),
        Route("/api/mcp/import/preview", endpoint=api_mcp_import_preview, methods=["POST"]),
    ]

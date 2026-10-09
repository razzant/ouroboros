"""HTTP surface for the shared MCP manager used by Settings and ToolRegistry."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict

from starlette.requests import Request
from starlette.responses import JSONResponse

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


async def api_mcp_status(request: Request) -> JSONResponse:
    """GET /api/mcp/status — masked status snapshot for the UI."""
    try:
        await asyncio.to_thread(_ensure_configured)
        payload = await asyncio.to_thread(get_manager().status_payload)
        return JSONResponse(payload)
    except Exception as exc:
        log.exception("api_mcp_status failed")
        return json_error(f"{type(exc).__name__}: MCP status failed")


async def api_mcp_refresh(request: Request) -> JSONResponse:
    """POST /api/mcp/refresh — refresh one or all servers."""
    try:
        body: Dict[str, Any] = await request_json_or(request, {})
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


async def api_mcp_test(request: Request) -> JSONResponse:
    """Probe the edited candidate with the same URL rehydration as Settings."""
    from ouroboros.gateway.settings import MCPSecretIdentityAmbiguous, _rehydrate_mcp_servers_payload

    try:
        body: Dict[str, Any] = await request_json_or(request, {})
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
    """Translate pasted client JSON into draft patches; no settings or transport I/O."""
    from ouroboros.mcp_import import preview_import

    headers = {"Cache-Control": "no-store"}
    body = await request_json_or(request, None)
    if not isinstance(body, dict) or not isinstance(body.get("text"), str) or not isinstance(body.get("servers"), list):
        return JSONResponse({"ok": False, "error": "Supply text and a servers list.", "entries": []},
                            status_code=400, headers=headers)
    return JSONResponse(preview_import(body["text"], body["servers"]), headers=headers)

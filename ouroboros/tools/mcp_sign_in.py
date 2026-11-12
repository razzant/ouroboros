"""``remote_mcp_sign_in``: the mind's action for a remote MCP server's sign-in (``ouroboros/mcp_oauth.py``).

A call to such a server that needs the owner's sign-in returns ``MCP_AUTH_REQUIRED`` at once.
Whether, when and where to ask the owner is the mind's judgment; this tool is its one action:
``begin`` issues the provider's sign-in link to give them, ``complete`` finishes the sign-in from
the address their browser stopped on (the builtin return page finishes it by itself when that
browser reaches this server), and ``sign_out`` forgets this installation's tokens and asks the
provider to revoke them. The owner's own app registration (a provider that registers no apps
itself) is entered on the server's card in Settings → Tools, never through a tool argument.

The server process owns the tool catalog (``docs/PERSISTENCE.md``), so a completion here hands
``mcp_signed_in`` to its host on the turn's event queue, and the host lists that server again
(``supervisor/events_worker_reports.py``). The receipt names the handoff, not a new listing.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict

from ouroboros.tools.registry import ToolContext, ToolEntry
from ouroboros.tools.tool_result import ToolResult, _publish_tool_result, publish_no_effect

log = logging.getLogger(__name__)


def _no_effect(ctx: ToolContext, status: str, code: str, text: str, **meta: Any) -> str:
    return publish_no_effect(ctx, ToolResult(status=status, code=code, text=f"⚠️ {text}", meta=meta))


def _receipt(ctx: ToolContext, body: Dict[str, Any]) -> str:
    return _publish_tool_result(ctx, ToolResult(status="ok", code="OK", text=json.dumps(body, ensure_ascii=False)))


def _hand_on_listing(ctx: ToolContext, server_id: str) -> str:
    """``queued_for_host`` (the supervisor lists the server as it drains the queue) or ``not_sent``."""
    event_queue = getattr(ctx, "event_queue", None)
    if event_queue is None:
        return "not_sent"
    try:
        event_queue.put_nowait({"type": "mcp_signed_in", "server_id": server_id})
        return "queued_for_host"
    except Exception:
        log.debug("mcp_signed_in not handed on", exc_info=True)
        return "not_sent"


def _mcp_sign_in(ctx: ToolContext, action: str = "", server: str = "", returned_url: str = "") -> str:
    from ouroboros import mcp_oauth
    from ouroboros.mcp_client import _run_async, canonical_server_id, ensure_configured_from_settings, get_manager

    if action not in ("begin", "complete", "sign_out"):
        return _no_effect(ctx, "error", "TOOL_ARG_ERROR", "TOOL_ARG_ERROR: action is begin, complete or sign_out.")
    try:
        returned = mcp_oauth.parse_return(returned_url) if action == "complete" else {}
        server_id = mcp_oauth.server_of(returned["state"]) if action == "complete" else canonical_server_id(server)
    except ValueError as exc:
        return _no_effect(ctx, "error", "TOOL_ARG_ERROR", f"TOOL_ARG_ERROR: {exc}.")
    ensure_configured_from_settings()
    cfg = get_manager().server_config(server_id)
    if cfg is None or not mcp_oauth.applies(cfg):
        return _no_effect(ctx, "unavailable", "MCP_UNAVAILABLE",
                          f"MCP_UNAVAILABLE: no configured remote MCP server {server_id!r} that signs in (an HTTP "
                          "or SSE server without saved headers or credentials).")
    try:
        if action == "begin":
            started = _run_async(lambda: mcp_oauth.begin(cfg))
            return _receipt(ctx, {**started, "note": (
                "Give the owner this link. After they sign in, the provider sends their browser to redirect_uri, "
                "and the builtin return page there finishes the sign-in when that browser reaches this server. "
                "Otherwise ask them for the address their browser stopped on and call complete with it. A later "
                "begin replaces this one.")})
        if action == "complete":
            done = _run_async(lambda: mcp_oauth.complete(cfg, **returned))
            return _receipt(ctx, {**done, "catalog_refresh": _hand_on_listing(ctx, cfg.id)})
        return _receipt(ctx, _run_async(lambda: mcp_oauth.sign_out(cfg)))
    except mcp_oauth.MCPAuthRequired as exc:  # refused before this step changed anything
        return _no_effect(ctx, "unavailable", "MCP_AUTH_REQUIRED", str(exc), sign_in_reason=exc.reason)
    # Any other failure (an unreachable provider) is the registry's ordinary tool error: its effect is unknown.


def get_tools() -> list:
    return [ToolEntry("remote_mcp_sign_in", {
        "name": "remote_mcp_sign_in",
        "description": (
            "Sign the owner in to a remote MCP server whose calls return MCP_AUTH_REQUIRED (an HTTP or SSE server "
            "without saved credentials), or sign them out. begin returns the provider's sign-in link to give the "
            "owner; usually their browser then comes back to this server and the sign-in finishes by itself. "
            "complete finishes it from the full address their browser stopped on, when the owner pastes it. "
            "sign_out forgets this installation's tokens and asks the provider to revoke them (revoked says what "
            "it confirmed). Starting a sign-in sends nothing to the owner; whether and when to ask is your call."
        ),
        "parameters": {"type": "object", "properties": {
            "action": {"type": "string", "enum": ["begin", "complete", "sign_out"]},
            "server": {"type": "string", "description": "For begin and sign_out: the server id from MCP_AUTH_REQUIRED."},
            "returned_url": {"type": "string",
                             "description": "For complete: the full address the owner's browser stopped on."},
        }, "required": ["action"], "additionalProperties": False},
    }, _mcp_sign_in, safety_policy="skip")]

"""Explicit, selected Settings-secret reads; passive projections stay masked."""

from __future__ import annotations

from starlette.requests import Request
from starlette.responses import JSONResponse

from ouroboros.config import SETTINGS_DEFAULTS, load_settings
from ouroboros.secret_masking import MASKED_SECRET_SETTING_KEYS, is_custom_secret_setting_key
from ouroboros.server_runtime import apply_runtime_provider_defaults


async def api_settings_secret(request: Request) -> JSONResponse:
    """Reveal one selected Settings secret through the ordinary owner API gate.

    Keep the passive settings projection masked. This read uses the same loaded
    settings, including environment fallback, and never saves or tests a key.
    """
    headers = {"Cache-Control": "no-store"}
    try:
        body = await request.json()
    except Exception:
        body = None
    if not isinstance(body, dict) or set(body) not in ({"key"}, {"mcp_server_id"}, {"mcp_server_id", "header_name"}):
        return JSONResponse({"error": "Select one key or MCP server.", "code": "invalid_secret_selector"},
                            status_code=400, headers=headers)
    selector = body.get("key", body.get("mcp_server_id"))
    if any(not isinstance(value, str) or not value.strip() for value in body.values()):
        return JSONResponse({"error": "The secret selector must be a nonempty string.",
                             "code": "invalid_secret_selector"}, status_code=400, headers=headers)
    settings, _, _ = apply_runtime_provider_defaults(load_settings())
    if "key" in body:
        if selector not in MASKED_SECRET_SETTING_KEYS and not (
            selector in settings and is_custom_secret_setting_key(
                selector, known_setting_keys=SETTINGS_DEFAULTS)
        ):
            return JSONResponse({"error": "That Settings secret is not available.",
                                 "code": "secret_not_found"}, status_code=404, headers=headers)
        value = settings.get(selector)
    else:
        from ouroboros.mcp_client import canonical_server_id, raw_server_id

        server_id = canonical_server_id(selector)
        servers = settings.get("MCP_SERVERS")
        matches = [entry for entry in servers if server_id and raw_server_id(entry) == server_id] if isinstance(servers, list) else []
        if len(matches) != 1:
            ambiguous = len(matches) > 1
            return JSONResponse({
                "error": "More than one saved MCP server uses this identity." if ambiguous else "The saved MCP server was not found.",
                "code": "MCP_ID_AMBIGUOUS_SECRET" if ambiguous else "secret_not_found",
            }, status_code=409 if ambiguous else 404, headers=headers)
        if "header_name" in body:
            from ouroboros.mcp_headers import find_saved_header

            status, value = find_saved_header(matches[0].get("headers", {}), body["header_name"])
            if status != "found" or not isinstance(value, str):
                missing = status == "missing"
                return JSONResponse({"error": "The selected saved header is missing or not uniquely readable.",
                                     "code": "secret_not_found" if missing else "mcp_header_unreadable"},
                                    status_code=404 if missing else 409, headers=headers)
        else:
            value = matches[0].get("auth_token")
    return JSONResponse({"value": str(value or "")}, headers=headers)

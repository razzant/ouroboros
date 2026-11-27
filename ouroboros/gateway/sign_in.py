"""Owner routes of skills' service sign-ins (``ouroboros/service_sign_in.py``): the card's sign-in row.

``POST /api/sign-in`` runs one ``action`` (``SignInRequest``): ``begin`` answers the provider page the
owner opens, ``complete`` takes the provider's return (from the builtin return page, or the address the
browser stopped on as ``returned_url``), ``cancel`` retires the identified attempt, ``disconnect`` forgets
its grant here; ``revoke`` forgets it once the provider confirmed every token, else keeps it (``revocation_not_confirmed``,
``revocation_not_asked``). ``begin`` answers for the ``registration`` shown (a client the host registers: the ``issuer`` shown), ``disconnect`` and ``revoke`` for the ``consent``
shown. The client actions edit the owner's clients.
A refused step is 200 with ``ok`` false
(``code``, ``reason``, ``error``), as ``/api/mcp/sign-in`` answers. ``GET /api/sign-in/status`` paints the
rows. Each step runs on its own loop in a worker thread, so a sign-in's lock never blocks this loop; no
answer carries a token, a secret or a verifier, and every error text is masked.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict

import httpx
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from ouroboros.gateway._helpers import json_error, request_base, request_json_or

log = logging.getLogger(__name__)

ACTIONS = ("begin", "complete", "cancel", "disconnect", "revoke", "client_save", "client_remove")


def _masked(text: str) -> str:
    from ouroboros import service_sign_in
    from ouroboros.secret_masking import redact_known_values

    return redact_known_values(str(text), service_sign_in.secret_values())


async def _step(body: Dict[str, Any], return_base: str) -> Dict[str, Any]:
    from ouroboros import oauth_engine
    from ouroboros import service_sign_in as sign_in

    action, skill, service, account = body["action"], str(body.get("skill") or ""), str(body.get("service") or ""), \
        str(body.get("account") or "")
    if action == "begin":
        return await sign_in.begin(skill, service, client=str(body.get("client") or ""), account=account,
                                   optional_scopes=[str(s) for s in body.get("optional_scopes") or []],
                                   scope_words=body.get("scope_words"), registration=str(body.get("registration") or ""),
                                   login_hint=str(body.get("login_hint") or ""), return_base=return_base,
                                   register_client=body.get("register_client") is True, shown_issuer=str(body.get("issuer") or ""))
    if action == "complete":
        returned = oauth_engine.parse_return(body["returned_url"]) if body.get("returned_url") else {
            key: body[key] for key in ("state", "code", "error", "error_description", "iss") if key in body}
        done = await sign_in.complete(returned, attempt_id=body.get("attempt_id") or "")
        if done.get("state") == "signed_in":  # a conversation that offered this sign-in hears it once
            from ouroboros.sign_in_offers import announce
            announce(sign_in._sign_in_data_dir(), done["skill"], done["service"], str(done.get("label") or ""))
        return done
    if action == "cancel":
        return await sign_in.cancel(skill, service, attempt_id=body.get("attempt_id") or "")
    if action in ("disconnect", "revoke"):
        return await sign_in.withdraw(skill, service, account, revoke_at_provider=action == "revoke",
                                      consent=str(body.get("consent") or ""))
    if action == "client_save":
        client = body.get("client")
        if not isinstance(client, dict) or not body.get("issuer"):
            raise ValueError("client_save needs issuer and a client object")
        return {"ok": True, "client": await sign_in.client_save(body["issuer"], client, endpoints=body.get("endpoints"))}
    return {"ok": True, "removed": await sign_in.client_remove(str(body.get("client") or ""))}


async def api_sign_in(request: Request) -> JSONResponse:
    """POST /api/sign-in — one step of a skill's sign-in to a service it declares."""
    from ouroboros import config, service_sign_in
    from ouroboros.gateway.schema import validate_ingress
    from ouroboros.gateway.sign_in_contracts import SignInRequest
    from ouroboros.oauth_engine import SignInRequired

    body = await request_json_or(request, None)
    if not isinstance(body, dict) or body.get("action") not in ACTIONS or validate_ingress(body, SignInRequest):
        return json_error(f"supply action {', '.join(ACTIONS)} with its fields", 400)
    data_dir = getattr(request.app.state, "drive_root", None) or config.DATA_DIR
    with service_sign_in.at_data_dir(data_dir):
        try:
            result = await asyncio.to_thread(asyncio.run, _step(body, request_base(request)))
            return JSONResponse({"ok": True, **result})
        except service_sign_in.SignInNotFound as exc:
            return json_error(_masked(str(exc)), 404)
        except SignInRequired as exc:
            attempt = getattr(exc, "attempt_id", "")
            return JSONResponse({"ok": False, "code": exc.code, "reason": exc.reason,
                                 "error": _masked(exc.detail or exc.reason),
                                 **({"attempt_id": attempt} if attempt else {})})
        except ValueError as exc:
            return json_error(_masked(str(exc)), 400)
        except httpx.HTTPError as exc:
            return JSONResponse({"ok": False, "code": "SIGN_IN_UNAVAILABLE", "reason": "provider_unreachable",
                                 "error": _masked(f"{type(exc).__name__}: {exc}")})
        except Exception as exc:
            log.warning("api_sign_in %s failed (%s)", body.get("action"), type(exc).__name__)
            return json_error(f"{type(exc).__name__}: sign-in step failed", 500)


async def api_sign_in_status(request: Request) -> JSONResponse:
    """GET /api/sign-in/status — every installed skill's declared services, accounts, pending sign-ins and clients."""
    from ouroboros import config, service_sign_in
    from ouroboros.oauth_engine import SignInRequired

    data_dir = getattr(request.app.state, "drive_root", None) or config.DATA_DIR
    with service_sign_in.at_data_dir(data_dir):
        try:
            payload = await asyncio.to_thread(service_sign_in.status, return_base=request_base(request))
        except SignInRequired as exc:
            return JSONResponse({"ok": False, "code": "SIGN_IN_UNAVAILABLE", "reason": exc.reason,
                                 "error": _masked(exc.detail or exc.reason)}, status_code=503)
    return JSONResponse(payload)


def sign_in_routes() -> list:
    return [Route("/api/sign-in", api_sign_in, methods=["POST"]),
            Route("/api/sign-in/status", api_sign_in_status, methods=["GET"])]

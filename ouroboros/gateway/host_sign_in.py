"""Host Service routes of skills' service sign-ins (``ouroboros/service_sign_in.py``): a skill's own tokens.

The skill is the one its ``x-skill-token`` names (installed, enabled, trusted code with executable review
for its current content hash: ``_assert_active_token``), never one the body names, so no skill reaches
another skill's grant whatever it holds. ``GET /sign-in/accounts`` lists its accounts per declared service;
``POST /sign-in/token`` answers a short-lived access token (one refresh under the account's lock when the
stored one is the ``rejected`` one, expired or valid less than ``min_valid`` seconds; the refresh token,
client secret and PKCE verifier never leave the host); ``POST /sign-in/handover`` takes over sign-ins the
skill already held (``service_sign_in.handover``) or reads one verified offer's saved receipt. A refresh runs on its own loop in a worker thread, so the
lock never blocks this loop. Every error text is masked with the live sign-in secrets; the host logs counts
and outcomes only. Beside ``host_service.py``; ``create_host_service_app`` mounts these routes.
"""

from __future__ import annotations

import asyncio
import logging
import pathlib
from typing import Any, Dict

import httpx
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from ouroboros.gateway.host_service import HostServiceAuthError, _authenticated, _json_error

log = logging.getLogger(__name__)

SIGN_IN_VERSION = 1
SIGN_IN_HANDOVER_VERSION = 2  # unknown returns, verified handover receipts and consent-bound requests
CARD_PLACE = "Extensions › Skills"  # where the owner signs in: the skill's card
_HINT = ("SIGN_IN_REQUIRED: {skill} needs the owner's sign-in to {label} for {account} ({reason}). "
         f"The owner can press Sign in again on its card in {CARD_PLACE}"
         " (to give the owner that card as a button here: send_links with place skill-sign-in/{skill}/{service}).")


def _masked(text: str) -> str:
    from ouroboros import service_sign_in
    from ouroboros.secret_masking import redact_known_values

    return redact_known_values(str(text), service_sign_in.secret_values())


def _busy() -> JSONResponse:
    """Another sign-in step held a lock this request waited for past its limit (``sign_in_busy``): temporary, and it confirms
    nothing this request may already have done."""
    return JSONResponse({"ok": False, "code": "SIGN_IN_UNAVAILABLE", "reason": "sign_in_busy"}, status_code=503)


async def _caller(request: Request, *, handover: bool = False):
    """``(skill, body, None)`` for an authenticated request with a JSON object body, else ``(.., .., refusal)``. Only a
    handover the rate limit turns away before its body is read says ``handover_started: false``: a fact of that request."""
    ctx = request.app.state.host_service_context
    try:
        skill, _payload = await _authenticated(ctx, request.headers.get("x-skill-token", ""))
    except HostServiceAuthError as exc:
        return "", {}, _json_error(str(exc), 403)
    if not ctx.rate_limiter.allow(f"{skill}:sign_in"):
        return skill, {}, JSONResponse({"ok": False, "error": "rate limit exceeded",
                                        **({"handover_started": False} if handover else {})}, status_code=429)
    body: Any = {}
    if request.method == "POST":
        try:
            body = await request.json()
        except Exception:
            body = None
        if not isinstance(body, dict):
            return skill, {}, _json_error("the body must be a JSON object", 400)
    return skill, body, None


def _view(skill: str, data_dir: pathlib.Path) -> list:
    from ouroboros import service_sign_in
    from ouroboros import service_sign_in_declaration as declaration
    from ouroboros.skill_loader import find_skill

    keep = ("key", "label", "subject", "verified", "state", "reason", "via", "consent", "legacy_generation", "connection_id")
    return [{"id": service.id, "label": service.label, "accounts": [
        {**{k: row[k] for k in keep if k in row}, "scope": " ".join(row["scope"]), "client_label": row["client"]["label"]}
        for row in service_sign_in.accounts(skill, service)]}
        for service in declaration.declared(find_skill(data_dir, skill)).values()]


async def api_sign_in_accounts(request: Request) -> JSONResponse:
    """List the authenticated skill's accounts. A store read failure is not an empty account list."""
    from ouroboros import service_sign_in
    from ouroboros.oauth_engine import SignInRequired

    skill, _body, refused = await _caller(request)
    if refused is not None:
        return refused
    data_dir = request.app.state.host_service_context.data_dir
    with service_sign_in.at_data_dir(data_dir):
        try:
            return JSONResponse({"ok": True, "services": await asyncio.to_thread(_view, skill, data_dir)})
        except SignInRequired as exc:
            return JSONResponse({"ok": False, "code": "SIGN_IN_UNAVAILABLE", "reason": exc.reason}, status_code=503)


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


async def api_sign_in_token(request: Request) -> JSONResponse:
    """``{service, account?, rejected?, required_scope?, min_valid?, connection_id?, consent?}`` → ``200 {ok, access_token,
    token_type, expires_at, scope, account: {key, label}, connection_id?, consent?}`` (an expected ``connection_id`` the account
    lacks: ``409 connection_changed``; an expected ``consent`` it no longer has, sent with explicit ``account`` and
    ``connection_id``: ``409 consent_changed``); ``409 SIGN_IN_REQUIRED`` with ``reason``, ``note`` (the
    skill's own words for that reason) and ``hint`` (one sentence for the mind); ``503 SIGN_IN_UNAVAILABLE``
    (the provider did not answer, a saved list cannot be read, or another sign-in step held the account's lock past its wait:
    ``sign_in_busy``; the grant is kept); ``404`` with ``reason`` and ``hint`` when there is
    no account to use (an undeclared service, an unknown account, none or several signed in)."""
    from ouroboros import service_sign_in
    from ouroboros.oauth_engine import SignInRequired
    from ouroboros.store import StoreBusy

    skill, body, refused = await _caller(request)
    if refused is not None:
        return refused
    service, account = str(body.get("service") or ""), str(body.get("account") or "")
    min_valid = body.get("min_valid", 0)
    if not service or not _number(min_valid) or any(
            body.get(key) is not None and not isinstance(body[key], str) for key in ("account", "rejected", "required_scope", "connection_id")):
        return _json_error("supply service, and account, rejected, required_scope, connection_id as strings, min_valid as seconds", 400)
    consent = body.get("consent")
    if "consent" in body and (not isinstance(consent, str) or not consent or not account or not body.get("connection_id")):
        return _json_error("consent must be a nonempty string with explicit account and connection_id", 400)
    ctx = request.app.state.host_service_context
    if not ctx._enter_inflight(skill):
        return _json_error("too many sign-in requests in flight", 429)
    try:
        with service_sign_in.at_data_dir(ctx.data_dir):
            try:
                answer = await asyncio.to_thread(asyncio.run, service_sign_in.token(
                    skill, service, account, rejected=body.get("rejected") or None,
                    required_scope=str(body.get("required_scope") or ""), min_valid=min_valid,
                    connection_id=str(body.get("connection_id") or ""), consent=consent))
                return JSONResponse({"ok": True, **answer})
            except service_sign_in.SignInNotFound as exc:
                return await asyncio.to_thread(_not_found, skill, service, exc)
            except SignInRequired as exc:
                return await asyncio.to_thread(_required_response, skill, service, account, exc)
            except StoreBusy:
                return _busy()
            except ValueError as exc:
                return _json_error(_masked(str(exc)), 400)
            except httpx.HTTPError as exc:
                log.info("sign-in token for %s/%s: provider unreachable (%s)", skill, service, type(exc).__name__)
                return JSONResponse({"ok": False, "code": "SIGN_IN_UNAVAILABLE", "reason": "provider_unreachable",
                                     "error": _masked(f"{type(exc).__name__}: {exc}")}, status_code=503)
    finally:
        ctx._leave_inflight(skill)


def _required_response(skill: str, service: str, account: str, exc: Any) -> JSONResponse:
    from ouroboros import service_sign_in

    if exc.reason in ("sign_in_storage_unavailable", "sign_in_busy"):  # temporary: the grant is kept
        return JSONResponse({"ok": False, "code": "SIGN_IN_UNAVAILABLE", "reason": exc.reason}, status_code=503)
    declared = service_sign_in.declared_service(skill, service)
    label = getattr(exc, "account_label", "") or account or "the account"
    note = declared.notes.get("refresh_rejected", "") if declared and exc.reason == "refresh_rejected" else ""
    fixes = {
        "provider_scope_mismatch": "Review permissions and sign in again on the skill's card. If the provider still returns extra permissions, choose a separate approved client.",
        "provider_scope_unconfirmed": "The provider's permissions are not known. Sign in again on the skill's card.",
        "provider_approval_required": "Choose a client approved for this provider and sign in again on the skill's card.",
        "account_selection_required": "Disconnect the other accounts on the skill's card before using this service.",
        "connection_changed": "This account is no longer the connection you expected: read GET /sign-in/accounts and follow its current connection_id.",
        "consent_changed": "The sign-in has changed since this operation began. Confirm a new operation with its current consent.",
    }
    hint = fixes.get(exc.reason) or (
        "The token cannot be shown to last that long. Ask for a shorter min_valid interval."
        if exc.reason == "min_valid_unmet" else _HINT.format(
            skill=skill, label=declared.label if declared else service, account=label, reason=exc.reason, service=service))
    return JSONResponse({"ok": False, "code": "SIGN_IN_REQUIRED", "reason": exc.reason, "service": service,
                         "account": getattr(exc, "account", "") or account, "note": note, "hint": hint}, status_code=409)


# What a skill is told, by ``SignInNotFound.reason``, when there is no account to hand a token from.
_NO_ACCOUNT = {
    "unknown_service": "This skill declares no such service; use a service id from GET /sign-in/accounts.",
    "unknown_account": "That account is not signed in here; use a key from GET /sign-in/accounts.",
    "account_required": "Several accounts are signed in; pass `account`, one of: {accounts}.",
    "not_signed_in": "The owner has not signed in to {label}. Ask them to sign in on the skill's card in {place} (a button to it here: "
                     "send_links with place skill-sign-in/{skill}/{service}); do not retry before they have.",
    "no_usable_account": "No signed-in account of {label} can be used now ({accounts}; state and reason: GET /sign-in/accounts). "
                         "The owner can sign in again on the skill's card in {place}.",
}


def _not_found(skill: str, service: str, exc: Exception) -> JSONResponse:
    """The 404 for no account to use: the finder's own reason, the accounts its read found and one next step; no second read."""
    from ouroboros import service_sign_in

    reason = getattr(exc, "reason", "") or "not_found"
    declared = service_sign_in.declared_service(skill, service)
    accounts = ", ".join(f"{key} ({name})" if name else key for key, name in getattr(exc, "accounts", ()))
    hint = _NO_ACCOUNT.get(reason, "No such sign-in; check the service and account in GET /sign-in/accounts.").format(
        label=declared.label if declared else service, accounts=accounts, place=CARD_PLACE, skill=skill, service=service)
    return JSONResponse({"ok": False, "error": _masked(str(exc)), "reason": reason, "hint": hint}, status_code=404)


async def api_sign_in_handover(request: Request) -> JSONResponse:
    """Import the skill's saved grants, with receipts scoped to an optional exact legacy generation, or read one verified
    offer's saved receipt without the provider (``receipt``).

    Wire and replay rules: CREATING_SKILLS, "Signing in to a service".
    """
    from ouroboros import service_sign_in
    from ouroboros.oauth_engine import SignInRequired
    from ouroboros.skill_loader import find_skill
    from ouroboros.store import StoreBusy

    skill, body, refused = await _caller(request, handover=True)
    if refused is not None:
        return refused
    if "receipt" in body:
        receipt, service = body["receipt"], body.get("service")
        if (set(body) != {"service", "receipt"} or not isinstance(service, str) or not service
                or not isinstance(receipt, dict) or set(receipt) != {"legacy_key", "legacy_generation"}
                or not all(isinstance(value, str) for value in receipt.values())):
            return _json_error("supply only service and receipt with legacy_key and legacy_generation", 400)
        with service_sign_in.at_data_dir(request.app.state.host_service_context.data_dir):
            try:
                answer = await asyncio.to_thread(asyncio.run, service_sign_in.read_handover_receipt(
                    skill, service, receipt["legacy_key"], receipt["legacy_generation"]))
            except service_sign_in.SignInNotFound as exc:
                return _json_error(_masked(str(exc)), 404)
            except SignInRequired as exc:
                return JSONResponse({"ok": False, "code": "SIGN_IN_UNAVAILABLE", "reason": exc.reason}, status_code=503)
            except StoreBusy:
                return _busy()
            except ValueError as exc:
                return _json_error(_masked(str(exc)), 400)
        return JSONResponse({"ok": True, "receipt": answer})
    clients, grants = body.get("clients") or [], body.get("grants") or []
    if not body.get("service") or not isinstance(clients, list) or not isinstance(grants, list) or not all(
            isinstance(item, dict) for item in (*clients, *grants)) or not isinstance(body.get("recover", False), bool):
        return _json_error("supply service, lists of client and grant objects, and recover as true or false", 400)
    data_dir = request.app.state.host_service_context.data_dir
    with service_sign_in.at_data_dir(data_dir):
        loaded = await asyncio.to_thread(find_skill, data_dir, skill)
        version = str(getattr(getattr(loaded, "manifest", None), "version", "") or "")
        try:
            answer: Dict[str, Any] = await asyncio.to_thread(asyncio.run, service_sign_in.handover(
                skill, str(body["service"]), clients_in=clients, grants=grants, version=version,
                recover=body.get("recover") is True))
        except service_sign_in.SignInNotFound as exc:
            return _json_error(_masked(str(exc)), 404)
        except (httpx.HTTPError, ValueError) as exc:
            return JSONResponse({"ok": False, "code": "SIGN_IN_UNAVAILABLE", "reason": "provider_unreachable",
                                 "error": _masked(f"{type(exc).__name__}: {exc}")}, status_code=503)
        except StoreBusy:  # the request may have started: recover or read its receipts
            return _busy()
        except (SignInRequired, OSError) as exc:  # a store its answers need could not be read or written: no receipts
            return JSONResponse({"ok": False, "code": "SIGN_IN_UNAVAILABLE",
                                 "reason": getattr(exc, "reason", "sign_in_storage_unavailable")}, status_code=503)
    log.info("sign-in handover for %s/%s: %s", skill, body["service"],
             [item.get("result") for item in answer.get("grants", [])])
    return JSONResponse(answer)


def host_sign_in_routes() -> list:
    return [Route("/sign-in/accounts", api_sign_in_accounts, methods=["GET"]),
            Route("/sign-in/token", api_sign_in_token, methods=["POST"]),
            Route("/sign-in/handover", api_sign_in_handover, methods=["POST"])]

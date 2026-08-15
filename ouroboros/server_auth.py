"""Minimal auth gate for non-localhost browser/server access."""

from __future__ import annotations

import base64
import hashlib
import hmac
import html
import ipaddress
import os
import pathlib
import secrets
import time
from http.cookies import SimpleCookie
from urllib.parse import parse_qs, unquote, urlsplit

from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, RedirectResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from ouroboros.config import load_settings
from ouroboros.remote_refusal_actions import (
    ACTION_AUTHENTICATE_OWNER,
    ACTION_CONFIGURE_NETWORK_PASSWORD,
)

NETWORK_PASSWORD_KEY = "OUROBOROS_NETWORK_PASSWORD"
AUTH_COOKIE_NAME = "ouroboros_auth"
_PUBLIC_HTTP_PATHS = {"/api/health", "/auth/login", "/auth/logout"}
_OWNER_CONNECTIONS_PATH = "/api/owner/connections"


def get_configured_network_password() -> str:
    raw = (os.environ.get(NETWORK_PASSWORD_KEY, "") or "").strip()
    if raw:
        return raw
    try:
        return str(load_settings().get(NETWORK_PASSWORD_KEY, "") or "").strip()
    except Exception:
        return ""


def is_loopback_host(host: str | None) -> bool:
    text = (host or "").strip().lower()
    if not text:
        return False
    # Normalize bracketed IPv6 literals (e.g. "[::1]" → "::1") so ipaddress can parse them.
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    if text in {"localhost", "127.0.0.1", "::1"}:
        return True
    try:
        return ipaddress.ip_address(text.split("%", 1)[0]).is_loopback
    except ValueError:
        return False


def validate_network_auth_configuration(bind_host: str) -> str | None:
    return None


def get_network_auth_startup_warning(bind_host: str) -> str | None:
    if is_loopback_host(bind_host):
        return None
    if get_configured_network_password():
        return None
    return (
        "Server is binding to a non-loopback host without OUROBOROS_NETWORK_PASSWORD. "
        "Access will stay open to the network until a password is configured."
    )


def _headers_map(scope: Scope) -> dict[str, str]:
    return {
        key.decode("latin-1").lower(): value.decode("latin-1")
        for key, value in scope.get("headers", [])
    }


# Sessions expire; a leaked cookie is no longer a permanent credential.
_AUTH_SESSION_TTL_SEC = 30 * 24 * 3600
_auth_secret_cache: bytes | None = None


def _auth_secret() -> bytes:
    """Server-side random HMAC key, persisted across restarts (0600).

    The old cookie was a bare ``sha256(password)`` — deterministic, never
    expiring, and forgeable by anyone who ever saw it. Sessions are now
    HMAC-signed with this key, so they cannot be minted without server state
    and they die with the key file (one re-login after key loss is accepted).
    """
    global _auth_secret_cache
    if _auth_secret_cache:
        return _auth_secret_cache
    try:
        from ouroboros import config as _cfg

        path = pathlib.Path(_cfg.DATA_DIR) / "state" / "auth_secret.key"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            text = path.read_text(encoding="utf-8").strip()
            if text:
                _auth_secret_cache = bytes.fromhex(text)
                return _auth_secret_cache
        secret = secrets.token_hex(32)
        path.write_text(secret + "\n", encoding="utf-8")
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
        _auth_secret_cache = bytes.fromhex(secret)
    except Exception:
        # No durable key (read-only data dir): fall back to a process-lifetime
        # key — sessions survive within the run and re-login after restart.
        _auth_secret_cache = secrets.token_bytes(32)
    return _auth_secret_cache


def _session_value(password: str, issued_at: int | None = None) -> str:
    issued = int(issued_at if issued_at is not None else time.time())
    pw_digest = hashlib.sha256(f"ouroboros-auth\0{password}".encode("utf-8")).hexdigest()
    signature = hmac.new(
        _auth_secret(), f"{issued}\0{pw_digest}".encode("utf-8"), hashlib.sha256
    ).hexdigest()
    return f"{issued}.{signature}"


def _cookie_session_valid(cookie_val: str, password: str) -> bool:
    issued_text, sep, _signature = (cookie_val or "").partition(".")
    if not sep:
        return False
    try:
        issued = int(issued_text)
    except (TypeError, ValueError):
        return False
    now = time.time()
    if issued > now + 300 or now - issued > _AUTH_SESSION_TTL_SEC:
        return False
    return hmac.compare_digest(cookie_val, _session_value(password, issued))


def _cookie_value(scope: Scope, cookie_name: str) -> str:
    cookie_header = _headers_map(scope).get("cookie", "")
    if not cookie_header:
        return ""
    jar = SimpleCookie()
    jar.load(cookie_header)
    morsel = jar.get(cookie_name)
    return morsel.value if morsel else ""


def _candidate_password(scope: Scope) -> str:
    headers = _headers_map(scope)
    direct = headers.get("x-ouroboros-password", "").strip()
    if direct:
        return direct

    auth = headers.get("authorization", "").strip()
    if not auth:
        return ""
    scheme, _, rest = auth.partition(" ")
    if scheme.lower() == "bearer":
        return rest.strip()
    if scheme.lower() == "basic":
        try:
            decoded = base64.b64decode(rest).decode("utf-8")
        except Exception:
            return ""
        _, _, password = decoded.partition(":")
        return password if password else decoded
    return ""


def _is_authenticated(scope: Scope, password: str) -> bool:
    cookie_val = _cookie_value(scope, AUTH_COOKIE_NAME)
    if cookie_val and _cookie_session_valid(cookie_val, password):
        return True
    candidate = _candidate_password(scope)
    return bool(candidate) and hmac.compare_digest(candidate, password)


def _scope_client_host(scope: Scope) -> str | None:
    client = scope.get("client")
    if not client:
        return None
    return client[0]


def _request_wants_html(scope: Scope) -> bool:
    if scope.get("method", "GET").upper() != "GET":
        return False
    headers = _headers_map(scope)
    accept = headers.get("accept", "")
    path = scope.get("path", "")
    return path == "/" or "text/html" in accept


def is_owner_connections_path(path: str) -> bool:
    """Match the bounded twice-decoded owner namespace and reject lookalikes."""

    text = str(path or "").strip()
    try:
        parsed = urlsplit(text)
        if parsed.scheme or parsed.netloc:
            text = parsed.path
    except ValueError:
        pass
    # Starlette and user agents may each decode one layer. Two bounded passes
    # align the auth and file-surface defense-in-depth predicates without an
    # open-ended decoder loop.
    text = unquote(unquote(text)).rstrip("/") or "/"
    return text == _OWNER_CONNECTIONS_PATH or text.startswith(
        _OWNER_CONNECTIONS_PATH + "/"
    )


def _build_next_url(scope: Scope) -> str:
    raw_path = scope.get("path", "/") or "/"
    query_string = scope.get("query_string", b"")
    if not query_string:
        return raw_path
    return f"{raw_path}?{query_string.decode('latin-1')}"


def _sanitize_next_url(value: str) -> str:
    text = (value or "").strip()
    if not text.startswith("/"):
        return "/"
    if text.startswith("//"):
        return "/"
    if any(ch in text for ch in ('"', "'", "<", ">", "\r", "\n", "\x00")):
        return "/"
    return text


def _login_page(next_url: str, error: str = "") -> str:
    safe_next = html.escape(_sanitize_next_url(next_url), quote=True)
    error_html = (
        f'<div style="margin-top:12px;color:#ef4444;font-size:14px">{error}</div>'
        if error
        else ""
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Ouroboros Login</title>
</head>
<body style="margin:0;background:#0d0b0f;color:#e2e8f0;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',system-ui,sans-serif;display:flex;min-height:100vh;align-items:center;justify-content:center;">
  <form method="post" action="/auth/login" style="width:min(360px,calc(100% - 32px));padding:24px;border:1px solid rgba(255,255,255,0.08);border-radius:18px;background:rgba(255,255,255,0.04);box-shadow:0 24px 60px rgba(0,0,0,0.35);">
    <h1 style="margin:0 0 8px;font-size:22px;">Ouroboros</h1>
    <p style="margin:0 0 16px;color:rgba(255,255,255,0.65);line-height:1.5;">This server is exposed beyond localhost. Enter the network password to continue.</p>
    <input type="hidden" name="next" value="{safe_next}">
    <label for="password" style="display:block;margin-bottom:8px;font-size:14px;">Password</label>
    <input id="password" name="password" type="password" autofocus required style="width:100%;box-sizing:border-box;padding:12px 14px;border-radius:12px;border:1px solid rgba(255,255,255,0.12);background:#15121a;color:#e2e8f0;">
    <button type="submit" style="margin-top:16px;width:100%;padding:12px 14px;border:none;border-radius:12px;background:#c93545;color:white;font-weight:600;cursor:pointer;">Unlock</button>
    {error_html}
  </form>
</body>
</html>"""


async def _handle_login(scope: Scope, receive: Receive, send: Send, password: str) -> None:
    request = Request(scope, receive)
    if request.method == "GET":
        response = HTMLResponse(_login_page(request.query_params.get("next", "/")), status_code=200)
        await response(scope, receive, send)
        return
    if request.method != "POST":
        response = JSONResponse({"error": "Method not allowed."}, status_code=405)
        await response(scope, receive, send)
        return

    content_type = request.headers.get("content-type", "")
    next_url = "/"
    submitted = ""
    is_json = "application/json" in content_type
    if is_json:
        payload = await request.json()
        submitted = str(payload.get("password", "") or "")
        next_url = _sanitize_next_url(str(payload.get("next", "/") or "/"))
    else:
        form = await request.form()
        submitted = str(form.get("password", "") or "")
        next_url = _sanitize_next_url(str(form.get("next", "/") or "/"))

    if not submitted or not hmac.compare_digest(submitted, password):
        if is_json:
            response = JSONResponse({"error": "Invalid password."}, status_code=401)
        else:
            response = HTMLResponse(_login_page(next_url, "Invalid password."), status_code=401)
        await response(scope, receive, send)
        return

    if is_json:
        response = JSONResponse({"ok": True, "next": next_url}, status_code=200)
    else:
        response = RedirectResponse(next_url, status_code=303)
    # Secure only when actually served over TLS: setting it on plain-http LAN
    # would make the browser drop the cookie and lock the user in a login loop.
    response.set_cookie(
        AUTH_COOKIE_NAME,
        _session_value(password),
        max_age=_AUTH_SESSION_TTL_SEC,
        httponly=True,
        samesite="lax",
        secure=str(scope.get("scheme") or "").lower() in {"https", "wss"},
    )
    await response(scope, receive, send)


async def _handle_logout(scope: Scope, receive: Receive, send: Send) -> None:
    next_url = "/"
    if scope.get("method", "GET").upper() == "POST":
        request = Request(scope, receive)
        content_type = request.headers.get("content-type", "")
        if "application/json" in content_type:
            payload = await request.json()
            next_url = _sanitize_next_url(str(payload.get("next", "/") or "/"))
        else:
            form = await request.form()
            next_url = _sanitize_next_url(str(form.get("next", "/") or "/"))
    else:
        params = parse_qs(scope.get("query_string", b"").decode("latin-1"))
        next_url = _sanitize_next_url((params.get("next") or ["/"])[0])

    response = RedirectResponse(next_url, status_code=303)
    response.delete_cookie(AUTH_COOKIE_NAME)
    await response(scope, receive, send)


class NetworkAuthGate:
    """Require a password for non-localhost access only when configured."""

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in {"http", "websocket"}:
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "") or "/"
        password = get_configured_network_password()
        if scope["type"] == "http" and is_owner_connections_path(path):
            if not password:
                response = JSONResponse(
                    {
                        "error": "Owner authentication is not configured.",
                        "error_code": "owner_auth_not_configured",
                        "action": ACTION_CONFIGURE_NETWORK_PASSWORD,
                    },
                    status_code=503,
                )
                await response(scope, receive, send)
                return
            if not _is_authenticated(scope, password):
                response = JSONResponse(
                    {
                        "error": "Owner authentication is required.",
                        "error_code": "owner_auth_required",
                        "action": ACTION_AUTHENTICATE_OWNER,
                    },
                    status_code=401,
                )
                await response(scope, receive, send)
                return
            await self.app(scope, receive, send)
            return

        if not password:
            await self.app(scope, receive, send)
            return

        # Owner-only loopback surfaces use the same signed browser session as
        # non-local access.  Handle login/logout before the ordinary loopback
        # bypass so Settings can establish that session without ever retaining
        # the submitted password in JavaScript.
        if scope["type"] == "http" and path == "/auth/login":
            await _handle_login(scope, receive, send, password)
            return
        if scope["type"] == "http" and path == "/auth/logout":
            await _handle_logout(scope, receive, send)
            return

        if is_loopback_host(_scope_client_host(scope)):
            await self.app(scope, receive, send)
            return

        if scope["type"] == "http" and path in _PUBLIC_HTTP_PATHS:
            await self.app(scope, receive, send)
            return

        if _is_authenticated(scope, password):
            await self.app(scope, receive, send)
            return

        if scope["type"] == "websocket":
            await send({"type": "websocket.close", "code": 4401})
            return

        if _request_wants_html(scope):
            response = HTMLResponse(_login_page(_build_next_url(scope)), status_code=401)
        else:
            response = JSONResponse({"error": "Authentication required."}, status_code=401)
        await response(scope, receive, send)

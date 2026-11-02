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
from typing import TYPE_CHECKING
from urllib.parse import parse_qs, urlencode

from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, PlainTextResponse, RedirectResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from ouroboros.browser_boundary import CROSS_ORIGIN_CODE, JSON_CONTENT_TYPE_CODE, browser_request_refused, json_body_refused
from ouroboros.config import load_settings
from ouroboros.standalone_page import themed_page

if TYPE_CHECKING:
    from ouroboros.server_web import ServedUI

NETWORK_PASSWORD_KEY = "OUROBOROS_NETWORK_PASSWORD"
AUTH_COOKIE_NAME = "ouroboros_auth"
# The widget frame bridge is static host code a route page loads from its opaque frame, which carries no session.
_PUBLIC_HTTP_PATHS = {"/api/health", "/auth/login", "/auth/logout", "/api/widget-bridge.js"}
_NO_STORE = {"Cache-Control": "no-store"}


def resolve_network_password(env_value: str | None, settings_loader) -> str:
    """The resolution order, pure: a non-blank environment value wins, else the settings value,
    else the empty password (a missing, blank or unreadable settings file included)."""
    raw = (env_value or "").strip()
    if raw:
        return raw
    try:
        return str(settings_loader().get(NETWORK_PASSWORD_KEY, "") or "").strip()
    except Exception:
        return ""


def get_configured_network_password() -> str:
    return resolve_network_password(os.environ.get(NETWORK_PASSWORD_KEY, ""), load_settings)


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


def _build_next_url(scope: Scope) -> str:
    raw_path = scope.get("path", "/") or "/"
    query_string = scope.get("query_string", b"")
    if not query_string:
        return raw_path
    return f"{raw_path}?{query_string.decode('latin-1')}"


def _sanitize_next_url(value: str) -> str:
    """The same-origin path a login returns to, else ``/``. A separately hosted interface sends
    its own page here; browsers read a backslash as a slash and drop tabs and newlines, so
    ``/\\host`` or ``/<TAB>/host`` would otherwise leave the origin once the session exists."""
    text = (value or "").strip()
    if not text.startswith("/") or text.startswith("//") or "\\" in text:
        return "/"
    if any(ch in text for ch in ('"', "'", "<", ">")) or any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in text):
        return "/"
    return text


# The form's own layout on the interface's tokens; each has a system-colour fallback (standalone_page).
_LOGIN_CSS = """
body { margin: 0; min-height: 100vh; display: flex; align-items: center; justify-content: center;
  background: var(--bg-primary, Canvas); color: var(--text-primary, CanvasText); font-family: var(--font-sans, system-ui); }
.gate { box-sizing: border-box; width: min(360px, calc(100% - 32px)); padding: 24px; border-radius: var(--radius-object);
  background: var(--surface-object); box-shadow: var(--elev-rest); }
.gate h1 { margin: 0 0 8px; font-size: var(--type-page, 24px); font-weight: 600; line-height: var(--line-title); }
.gate p { margin: 0 0 16px; color: var(--text-secondary); font-size: var(--type-body); line-height: var(--line-body); }
.gate label { color: var(--text-primary); font-size: var(--type-body); }
.gate .btn { width: 100%; margin-top: 16px; }
.gate .gate-error { margin: 12px 0 0; color: var(--status-error-fg); }
"""


def _builtin_web_dir(served_ui: ServedUI | None) -> pathlib.Path:
    """The builtin interface tree, whatever ``OUROBOROS_UI_DIR`` selects: the builtin form wears only its files."""
    if served_ui is not None:
        return served_ui.builtin_dir
    from ouroboros.server_web import resolve_web_dir

    return resolve_web_dir(pathlib.Path(__file__).resolve().parents[1])


def _appearance_meta(served_ui: ServedUI | None, request: Request) -> str:
    """The form names the installation's appearance scope as the interface's own pages do."""
    return served_ui.appearance_meta(request.headers) if served_ui is not None else ""


def _login_page(next_url: str, error: str = "", web_dir: pathlib.Path | None = None, head: str = "") -> str:
    """The builtin form. It answers before any asset behind the gate may load, so it inlines the builtin
    appearance owner and tokens: this browser's saved Light / Dark / System choice, else the device's."""
    safe_next = html.escape(_sanitize_next_url(next_url), quote=True)
    error_html = f'\n  <p class="gate-error" role="alert">{html.escape(error)}</p>' if error else ""
    form = f"""<form class="gate" method="post" action="/auth/login">
  <h1>Ouroboros</h1>
  <p>This server is exposed beyond localhost. Enter the network password to continue.</p>
  <input type="hidden" name="next" value="{safe_next}">
  <div class="ui-field">
    <label for="password">Password</label>
    <input id="password" class="ui-control" name="password" type="password" autofocus required>
  </div>
  <button type="submit" class="btn btn-primary">Unlock</button>{error_html}
</form>"""
    return themed_page(web_dir or _builtin_web_dir(None), "Ouroboros Login", form, _LOGIN_CSS, head)


async def _handle_login(scope: Scope, receive: Receive, send: Send, password: str,
                        served_ui: ServedUI | None = None) -> None:
    request = Request(scope, receive)
    web_dir = _builtin_web_dir(served_ui)
    head = _appearance_meta(served_ui, request)
    if request.method == "GET":
        page = (served_ui.custom_page("auth/index.html")
                if served_ui is not None and request.query_params.get("builtin") != "1" else None)
        response = HTMLResponse(page if page is not None else _login_page(request.query_params.get("next", "/"),
                                                                          web_dir=web_dir, head=head),
                                headers=_NO_STORE)
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
            response = HTMLResponse(_login_page(next_url, "Invalid password.", web_dir, head), status_code=401,
                                    headers=_NO_STORE)
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

    def __init__(self, app: ASGIApp, *, served_ui: ServedUI | None = None):
        from ouroboros.gateway.widget_assets import WidgetAssetLeases

        self.app = app
        self.served_ui = served_ui
        self.widget_assets = WidgetAssetLeases()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in {"http", "websocket"}:
            await self.app(scope, receive, send)
            return

        password = get_configured_network_password()
        from ouroboros.gateway.widget_assets import (ASSET_PREFIX, RESOURCE_PREFIX, widget_asset_response,
                                                     widget_resource_response)

        epoch = hashlib.sha256(password.encode()).hexdigest()
        self.widget_assets.password_changed(epoch)
        scope["widget_asset_leases"] = self.widget_assets
        scope["widget_asset_epoch"] = epoch
        cookie = _cookie_value(scope, AUTH_COOKIE_NAME)
        owner = cookie if cookie and _cookie_session_valid(cookie, password) else _candidate_password(scope) or "local"
        scope["widget_asset_owner"] = hashlib.sha256(owner.encode()).hexdigest()
        active = bool(password) and not is_loopback_host(_scope_client_host(scope))
        path = scope.get("path", "") or "/"
        if scope["type"] == "http" and path == "/api/widgets/asset-leases":
            scope["widget_asset_admission"] = self.widget_assets.admit(scope["widget_asset_owner"])
        if scope["type"] == "http" and path.startswith(ASSET_PREFIX):
            response = await widget_asset_response(Request(scope, receive), self.widget_assets)
            await response(scope, receive, send)
            return
        if scope["type"] == "http" and path.startswith(RESOURCE_PREFIX):
            # The route lease is this request's whole authority: no session, boundary or password applies.
            await widget_resource_response(scope, receive, send, self.widget_assets, self.app)
            return
        # The browser boundary comes before every handler, the login form and the password check.
        if browser_request_refused(scope):
            if scope["type"] == "websocket":
                await send({"type": "websocket.close", "code": 4403})
            else:
                await JSONResponse({"error": "Requests from another site's page are not accepted.",
                                    "code": CROSS_ORIGIN_CODE}, status_code=403)(scope, receive, send)
            return
        if scope["type"] == "http":
            refused, receive = await json_body_refused(scope, receive)
            if refused:
                await JSONResponse({"error": "Send a JSON body with Content-Type: application/json.",
                                    "code": JSON_CONTENT_TYPE_CODE}, status_code=415)(scope, receive, send)
                return
        if scope["type"] == "http" and path in {"/auth/login", "/auth/logout"}:
            # These names belong to the gate even when it is inactive: a selected
            # asset named login/logout must not become a different operation.
            if not active:
                await PlainTextResponse("Not Found", status_code=404)(scope, receive, send)
            elif path == "/auth/login":
                await _handle_login(scope, receive, send, password, self.served_ui)
            else:
                self.widget_assets.revoke_owner(scope["widget_asset_owner"])
                await _handle_logout(scope, receive, send)
            return

        if not active:
            await self.app(scope, receive, send)
            return

        if scope["type"] == "http" and (path in _PUBLIC_HTTP_PATHS or (
                (path == "/auth" or path.startswith("/auth/")) and scope.get("method") in {"GET", "HEAD"})):
            await self.app(scope, receive, send)
            return

        if _is_authenticated(scope, password):
            await self.app(scope, receive, send)
            return

        if scope["type"] == "websocket":
            await send({"type": "websocket.close", "code": 4401})
            return

        if _request_wants_html(scope):
            builtin_setup = path == "/onboarding" and Request(scope).query_params.get("builtin") == "1"
            custom = (self.served_ui.custom_page("auth/index.html")
                      if self.served_ui is not None and not builtin_setup else None)
            next_url = _sanitize_next_url(_build_next_url(scope))
            if custom is not None:
                response = RedirectResponse("/auth/login?" + urlencode({"next": next_url}), status_code=303,
                                            headers=_NO_STORE)
            else:
                response = HTMLResponse(_login_page(next_url, web_dir=_builtin_web_dir(self.served_ui),
                                                    head=_appearance_meta(self.served_ui, Request(scope))),
                                        status_code=401, headers=_NO_STORE)
        else:
            response = JSONResponse({"error": "Authentication required."}, status_code=401)
        await response(scope, receive, send)

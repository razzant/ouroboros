"""One generation's selected interface and optional complete setup/login views.

``OUROBOROS_UI_DIR`` selects ``/`` and ``/static``; a directory may also provide
``onboarding.html`` and ``auth/index.html``. Its ``auth/`` is a public login asset
bundle. Missing optional pages fall back whole-page: the builtin wizard's graph
under ``/onboarding/static`` and the optional author kit always remain builtin.
Empty selects builtin, ``none`` serves no main interface, and a trusted directory
is published as-is. ``ServedUI.apply`` pins the choice once per server generation.
"""

from __future__ import annotations

import hashlib
import importlib.util
import logging
import os
import pathlib
from collections.abc import Mapping
from typing import Any, NamedTuple

from starlette.responses import FileResponse, HTMLResponse, PlainTextResponse
from starlette.routing import Mount
from starlette.staticfiles import StaticFiles

log = logging.getLogger(__name__)

UI_DIR_SETTING = "OUROBOROS_UI_DIR"
# Response-only ownership marker of the builtin index (never written into web/index.html):
# a page carrying it was served by the core itself and follows that core's served-UI facts.
UI_OWNER_MARKER = '<meta name="ouroboros-ui-owner" content="core">'
# Response-only too: the installation's appearance scope, which web/theme.js reads before any style.
APPEARANCE_SCOPE_META = "ouroboros-appearance-scope"
TELEGRAM_MINIAPP_HEADER = "X-Ouroboros-Telegram-MiniApp"
_NO_STORE = {"Cache-Control": "no-store"}
_NO_MAIN_UI_PAGE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><title>Ouroboros</title></head>
<body><h1>No main interface is served here</h1>
<p>This Ouroboros core runs with <code>OUROBOROS_UI_DIR=none</code>: its interface is hosted
separately. The core still answers <code>/api</code>, <code>/ws</code>, <code>/auth</code> and
<code>/onboarding</code>.</p></body></html>"""


def resolve_web_dir(repo_dir: pathlib.Path) -> pathlib.Path:
    repo_web_dir = repo_dir / "web"
    if repo_web_dir.exists():
        return repo_web_dir

    spec = importlib.util.find_spec("web")
    origin = getattr(spec, "origin", None) if spec else None
    if origin:
        package_dir = pathlib.Path(origin).resolve().parent
        if package_dir.exists():
            return package_dir

    return repo_web_dir


def read_author_kit_assets(repo_dir: pathlib.Path) -> dict[str, str]:
    """Read the installed portable controls for an opt-in author page.

    Extension handlers pass ``request.app.state.repo_dir`` so in-process and
    isolated route workers read the same serving tree. Authors return these
    texts through their own route or embed them in their HTML; this helper
    creates no endpoint, cache, state or authentication exception. Read at
    mount, not registration, so a new page receives the installed UI source.
    The kit is the core's builtin tree whatever ``OUROBOROS_UI_DIR`` selects.
    """
    web_dir = resolve_web_dir(pathlib.Path(repo_dir))
    return {
        "css": (web_dir / "ui.css").read_text(encoding="utf-8"),
        "javascript": (web_dir / "modules" / "ui_primitives.js").read_text(encoding="utf-8"),
    }


class NoCacheStaticFiles:
    """Wrap StaticFiles to add Cache-Control: no-cache headers."""

    def __init__(self, **kwargs):
        self._app = StaticFiles(**kwargs)

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            async def send_with_no_cache(message):
                if message["type"] == "http.response.start":
                    headers = [(k, v) for k, v in message.get("headers", []) if k.lower() != b"cache-control"]
                    headers.append((b"cache-control", b"no-cache, must-revalidate"))
                    message = {**message, "headers": headers}
                await send(message)

            await self._app(scope, receive, send_with_no_cache)
        else:
            await self._app(scope, receive, send)


def appearance_scope(drive_root: Any) -> str:
    """The installation's appearance scope: a digest of its own identity, the same on every port and
    another for every other installation (a copied or moved data root is one). ``""`` until the
    identity is minted, then kept through a record that fails to read again. Logical separation,
    never a secret: the browser keeps the choice in a cookie of this name, which reaches every
    server on that host."""
    from ouroboros.install_identity import recorded_id

    identity = recorded_id(drive_root) if drive_root else ""
    return hashlib.sha256(f"ouroboros appearance scope\0{identity}".encode()).hexdigest()[:16] if identity else ""


def _core_owned(html: str, facts: str = "") -> str:
    """Stamp the ownership marker and the response's other ``facts`` first in ``<head>`` (browsers hoist a leading meta there)."""
    if "<head>" in html:
        return html.replace("<head>", "<head>" + UI_OWNER_MARKER + facts, 1)
    return UI_OWNER_MARKER + facts + html


def make_index_page(web_dir: pathlib.Path, head=None):
    """The builtin index: core-owned, so marked and never cached across a selection change.
    ``head(request)`` names the response's further leading metas (the appearance scope)."""
    async def index_page(_request) -> HTMLResponse:
        index = web_dir / "index.html"
        if index.exists():
            html = index.read_bytes().decode("utf-8")  # the checkout's exact bytes, line endings included
            if _request.headers.get(TELEGRAM_MINIAPP_HEADER) == "1":
                # The authenticated sidecar proxy already stamps this presentation
                # hint. Bootstrap navigates to this document, so it needs its own SDK.
                html = html.replace("<html", '<html data-ouroboros-host="telegram"', 1)
                html = html.replace("<head>", '<head><script async src="https://telegram.org/js/telegram-web-app.js"></script>', 1)
            return HTMLResponse(_core_owned(html, head(_request) if head else ""), headers=_NO_STORE)
        return HTMLResponse("<html><body><h1>Ouroboros — web/ not found</h1></body></html>", status_code=404)

    return index_page


class UiSelection(NamedTuple):
    """One generation's applied main interface: what ``/`` and ``/static`` serve."""

    source: str  # "builtin" | "directory" | "none"
    root: pathlib.Path | None = None  # the served tree; None when no main interface is served
    error: str = ""  # why a requested directory was not applied (the builtin interface serves)

    def state(self) -> dict[str, str]:
        """``StateResponse.served_ui``: the applied source, not the request or an asset digest."""
        return {"source": self.source, **({"error": self.error} if self.error else {})}


def _same_or_inside(path: pathlib.Path, root: pathlib.Path) -> bool:
    try:
        path.resolve().relative_to(root)
        return True
    except (OSError, RuntimeError, ValueError):
        return False


def select_ui(value: Any, builtin_dir: pathlib.Path, *, protected: tuple[pathlib.Path, ...] = ()) -> UiSelection:
    """Resolve one ``OUROBOROS_UI_DIR`` value to the interface that is actually served.

    An explicit ``none`` is honored as given. A directory that cannot serve, or whose
    publication would expose the core's own data or source tree, leaves the builtin
    interface in place with the reason; a directory that physically is the builtin
    tree is the builtin interface. The directory itself is trusted, not sandboxed.
    """
    text = str(value or "").strip()
    builtin = UiSelection("builtin", builtin_dir)
    if not text:
        return builtin
    if text.lower() == "none":
        return UiSelection("none")

    def refused(reason: str) -> UiSelection:
        return builtin._replace(error=f"{UI_DIR_SETTING} {text!r} was not applied: {reason}")

    path = pathlib.Path(text).expanduser()
    if not path.is_absolute():
        return refused("expected an absolute directory, 'none' or an empty value")
    try:
        root = path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        return refused(f"the directory cannot be resolved ({type(exc).__name__})")
    if not root.is_dir():
        return refused(f"{root} is not a directory")
    try:
        if os.path.samefile(root, builtin_dir):
            return builtin
    except OSError:
        pass
    if not (root / "index.html").is_file():
        return refused(f"{root} has no index.html")
    for owned in protected:
        if _same_or_inside(owned, root):
            return refused(f"{root} contains the core's own tree {owned}, which /static would publish")
    return UiSelection("directory", root)


class _Generation(NamedTuple):
    selection: UiSelection
    static: Any  # the ASGI app serving /static, None without a main interface
    auth_static: Any = None  # the selected directory's intentionally public auth/ bundle
    drive_root: Any = None  # whose installation identity names the appearance scope


class ServedUI:
    """The main interface of this server generation, served at ``/`` and ``/static``.

    Routes are built at import with the builtin selection; the lifespan applies the
    saved choice once. One immutable generation record answers serving, the builtin
    marker and ``StateResponse.served_ui`` together, so they never disagree.
    """

    def __init__(self, repo_dir: pathlib.Path):
        self.repo_dir = pathlib.Path(repo_dir)
        self.builtin_dir = resolve_web_dir(self.repo_dir)
        self.builtin_dir.mkdir(parents=True, exist_ok=True)
        self._builtin_static = NoCacheStaticFiles(directory=str(self.builtin_dir))
        self._builtin_page = make_index_page(self.builtin_dir, head=lambda request: self.appearance_meta(request.headers))
        self._current = _Generation(UiSelection("builtin", self.builtin_dir), self._builtin_static)

    @property
    def selection(self) -> UiSelection:
        return self._current.selection

    def appearance_meta(self, headers: Mapping[str, str]) -> str:
        """The appearance scope's meta for a builtin entry page (``/``, ``/onboarding``, the network login form), else ``""``.

        Offered only while this generation serves the builtin interface, so every page of one origin keeps the
        choice in one place, and never to the Telegram Mini App: its proxy origin does not move with the port, and
        its connecting page reads that origin's own store."""
        current = self._current
        if current.selection.source != "builtin" or headers.get(TELEGRAM_MINIAPP_HEADER) == "1":
            return ""
        scope = appearance_scope(current.drive_root)
        return f'<meta name="{APPEARANCE_SCOPE_META}" content="{scope}">' if scope else ""

    def mounts(self) -> list[Mount]:
        """Public selected login assets, builtin recovery resources, then selected main assets."""
        return [
            Mount("/auth", app=self.auth_assets, name="auth_assets"),
            Mount("/onboarding/static", app=self._builtin_static, name="onboarding_static"),
            Mount("/static", app=self.static, name="static"),
        ]

    def apply(self, app: Any, settings: Mapping[str, Any]) -> UiSelection:
        """Apply this generation's saved or environment choice and attach it to ``app.state``."""
        from ouroboros.config import SETTINGS_PATH
        from ouroboros.server_process import record_applied_restart_settings

        requested = str(settings.get(UI_DIR_SETTING) or "")
        drive_root = getattr(getattr(app, "state", None), "drive_root", None)
        try:
            owned = (drive_root, self.repo_dir, pathlib.Path(SETTINGS_PATH).parent)  # data, source, credentials
            protected = tuple(pathlib.Path(p).resolve() for p in owned if p)
            selection = select_ui(requested, self.builtin_dir, protected=protected)
            static = None
            auth_static = None
            if selection.source == "builtin":
                static = self._builtin_static
            elif selection.source == "directory":
                static = NoCacheStaticFiles(directory=str(selection.root))
                auth_static = NoCacheStaticFiles(directory=str(selection.root / "auth"), check_dir=False)
        except Exception as exc:  # the core, its API and its controls start whatever the selection
            selection = UiSelection("builtin", self.builtin_dir,
                                    f"{UI_DIR_SETTING} {requested!r} was not applied: {type(exc).__name__}: {exc}")
            static = self._builtin_static
            auth_static = None
        self._current = _Generation(selection, static, auth_static, drive_root)
        app.state.served_ui = self
        record_applied_restart_settings({UI_DIR_SETTING: requested})
        if selection.error:
            log.error("%s; serving the builtin interface", selection.error)
        else:
            log.info("Main interface: %s%s", selection.source,
                     f" ({selection.root})" if selection.source == "directory" else "")
        return selection

    def custom_page(self, relative_path: str) -> str | None:
        """Read an optional whole page from the applied directory, never from saved intent.

        The two callers name onboarding.html or auth/index.html. Read before sending
        headers so a known read failure can still choose a complete builtin page.
        Missing is ordinary; unreadable or invalid UTF-8 is a diagnosed fallback.
        """
        selection = self._current.selection
        if selection.source != "directory":
            return None
        path = selection.root / relative_path
        try:
            return path.read_bytes().decode("utf-8")
        except FileNotFoundError:
            return None
        except (OSError, UnicodeError) as exc:
            log.warning("Could not read selected interface page %s; using builtin: %s", path, exc)
            return None

    async def auth_assets(self, scope, receive, send):
        """Serve only the selected login bundle; login/logout are reserved for the auth gate."""
        current = self._current
        app = current.auth_static
        if app is None or not (current.selection.root / "auth").is_dir():
            app = PlainTextResponse("Not Found", status_code=404)
        await app(scope, receive, send)

    async def index_page(self, request):
        current = self._current
        if current.selection.source == "builtin":
            return await self._builtin_page(request)
        if current.selection.source == "directory":
            index = current.selection.root / "index.html"
            if index.is_file():
                return FileResponse(str(index), media_type="text/html", headers=_NO_STORE)
            return HTMLResponse("<html><body><h1>The selected interface has no index.html</h1></body></html>",
                                status_code=404, headers=_NO_STORE)
        return HTMLResponse(_NO_MAIN_UI_PAGE, status_code=404, headers=_NO_STORE)

    async def static(self, scope, receive, send):
        app = self._current.static
        if app is None:
            app = PlainTextResponse("Not Found", status_code=404)
        await app(scope, receive, send)

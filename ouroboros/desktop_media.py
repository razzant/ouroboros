"""Grant the desktop's microphone request only on its own loopback page.

WebKitGTK denies UserMediaPermissionRequest by default. The shell installs its
handler before the first load; the request must ask for audio alone and the
view must still be at the exact scheme/host/port the launcher started. Camera,
display capture, unrelated origins and other permission classes stay denied.
The Windows and macOS WebViews keep their native permission prompts.

GTK's UserMediaPermissionRequest exposes media flags but no requesting-frame
origin. WebKit's Permissions Policy controls subframes; the app must not delegate
microphone permission to foreign embedded content. The view's current URI is the
native origin witness, never a prefix match or a client-provided URL.
"""

from __future__ import annotations

import ipaddress
import logging
import os
import sys
from urllib.parse import urlsplit

log = logging.getLogger("launcher.media")

# Includes paths changed by PyInstaller's GI runtime hooks. The packaged shell
# records them before the bootloader; a plain source launch records them here.
LINUX_DESKTOP_KEYS = (
    "LD_LIBRARY_PATH", "LD_LIBRARY_PATH_ORIG", "XDG_DATA_DIRS", "PYWEBVIEW_GUI",
    "GI_TYPELIB_PATH", "GIO_MODULE_DIR", "GIO_EXTRA_MODULES", "GSETTINGS_SCHEMA_DIR",
    "GTK_DATA_PREFIX", "GTK_EXE_PREFIX", "GTK_PATH", "GTK_IM_MODULE_FILE",
    "GDK_PIXBUF_MODULE_FILE", "GDK_PIXBUF_MODULEDIR",
    "GST_PLUGIN_SYSTEM_PATH_1_0", "GST_PLUGIN_SCANNER",
    "PANGO_LIBDIR", "PANGO_SYSCONFDIR", "GST_PLUGIN_PATH", "GST_PLUGIN_SYSTEM_PATH",
    "GST_REGISTRY", "GST_REGISTRY_FORK",
)
_HOST_PREFIX = "OUROBOROS_DESKTOP_HOST_"
_HOST_SAVED = "OUROBOROS_DESKTOP_ENV_SAVED"


def host_environment(base=None) -> dict[str, str]:
    """Restore caller GUI/loader values in a copy, never in the running shell.

    The saved state survives source exec but is consumed for host children, so
    server tools and Playwright start with the caller's environment. No global
    swap can race a WebKit helper or another browser-opening thread.
    """
    env = dict(os.environ if base is None else base)
    if env.pop(_HOST_SAVED, None) == "1":
        for key in LINUX_DESKTOP_KEYS:
            saved = env.pop(_HOST_PREFIX + key)  # incomplete custody is an error
            if saved.startswith("1"):
                env[key] = saved[1:]
            elif saved == "":
                env.pop(key, None)
            else:
                raise ValueError(f"Invalid saved desktop environment: {key}")
    return env


def route_gtk_browser(backend, opener) -> None:
    """pywebview 5.4's GTK _blank navigation uses this module-local reference.

    Replace that reference before windows exist; never patch stdlib webbrowser
    or swap the GUI process environment around an external-app launch.
    """
    if os.environ.get(_HOST_SAVED) == "1" and str(getattr(backend, "__name__", "")).endswith('.gtk'):
        from types import SimpleNamespace

        backend.webbrowser = SimpleNamespace(open=opener)


def loopback_origin(url: str):
    """Keep scheme, host and port because another local service is another origin."""
    try:
        parsed = urlsplit(url)
        host = parsed.hostname
        if parsed.scheme not in ("http", "https") or not host or parsed.username or parsed.password:
            return None
        if host != "localhost" and not ipaddress.ip_address(host).is_loopback:
            return None
        return parsed.scheme, host, parsed.port if parsed.port is not None else (443 if parsed.scheme == "https" else 80)
    except (TypeError, ValueError):
        return None


def handle_permission_request(view, request, *, origin, media_type) -> bool:
    """Resolve every permission explicitly so a later handler cannot broaden it."""
    allowed = False
    try:
        expected = loopback_origin(origin)
        allowed = bool(expected and loopback_origin(view.get_uri()) == expected
                       and isinstance(request, media_type)
                       and request.get_property("is-for-audio-device")
                       and not request.get_property("is-for-video-device"))
        # Display capture exists on newer GTK. Older libraries cannot request it.
        if allowed and request.find_property("is-for-display-device"):
            allowed = not request.get_property("is-for-display-device")
    except Exception:
        allowed = False
        log.debug("Cannot establish microphone permission facts", exc_info=True)
    if allowed:
        request.allow()
    else:
        request.deny()
    return True


def bind_media_view(view, origin, media_type):
    """Return a disposer for one native view's signal registration."""
    signal = view.connect("permission-request", lambda widget, request: handle_permission_request(
        widget, request, origin=origin, media_type=media_type))

    def dispose():
        if view.handler_is_connected(signal):
            view.disconnect(signal)

    return dispose


def attach_media_permissions(window, origin):
    """Attach before show, when GTK's native widget exists but no page has loaded."""
    if sys.platform != "linux" or not loopback_origin(origin):
        return window
    disposers = []

    def before_show():
        # Import the backend that pywebview selected; no second GTK initialization.
        gui = getattr(window, "gui", None)
        if not str(getattr(gui, "__name__", "")).endswith(".gtk"):
            return
        try:
            webkit = gui.webkit
            pending = [window.native]
            while pending:
                widget = pending.pop()
                if isinstance(widget, webkit.WebView):
                    disposers.append(bind_media_view(widget, origin, webkit.UserMediaPermissionRequest))
                else:
                    pending.extend(getattr(widget, "get_children", lambda: [])())
            if not disposers:
                log.warning("No GTK WebView found for microphone permissions")
        except Exception:
            log.warning("GTK microphone permission handler could not be attached", exc_info=True)

    def closed():
        for dispose in disposers:
            try:
                dispose()
            except Exception:
                log.debug("GTK view already destroyed", exc_info=True)
        disposers.clear()
        window.events.before_show -= before_show
        window.events.closed -= closed

    window.events.before_show += before_show
    window.events.closed += closed
    return window


def prepare_linux_bundle(bundle_dir) -> None:
    """Restore the shipped GTK resource paths after frozen GI runtime hooks.

    The packaged entry saves caller values before the loader and GI hooks.
    The source handoff inherits the GUI environment and this cwd; server and
    external-app launches use host_environment and their own cwd instead.
    Source installs without a native bundle keep their ordinary backend choice.
    """
    import os
    from pathlib import Path

    root = Path(bundle_dir) / "linux-gtk"
    if not (root / "manifest.json").is_file():
        return
    if os.environ.get(_HOST_SAVED) != "1":
        for key in LINUX_DESKTOP_KEYS:
            os.environ[_HOST_PREFIX + key] = "1" + os.environ[key] if key in os.environ else ""
        os.environ[_HOST_SAVED] = "1"
    lib = root / "lib/x86_64-linux-gnu"
    paths = {
        "GI_TYPELIB_PATH": lib / "girepository-1.0", "GIO_MODULE_DIR": lib / "gio/modules",
        "GIO_EXTRA_MODULES": lib / "gio/modules", "GSETTINGS_SCHEMA_DIR": root / "share/glib-2.0/schemas",
        "GTK_DATA_PREFIX": root, "GTK_EXE_PREFIX": root, "GTK_PATH": lib / "gtk-3.0",
        "GTK_IM_MODULE_FILE": lib / "gtk-3.0/3.0.0/immodules.cache",
        "GDK_PIXBUF_MODULE_FILE": lib / "gdk-pixbuf-2.0/2.10.0/loaders.cache",
        "GST_PLUGIN_SYSTEM_PATH_1_0": lib / "gstreamer-1.0",
        "GST_PLUGIN_SCANNER": lib / "gstreamer1.0/gstreamer-1.0/gst-plugin-scanner",
    }
    os.environ.update({key: str(path) for key, path in paths.items()})
    share = str(root / "share")
    existing = os.environ.get("XDG_DATA_DIRS", "/usr/local/share:/usr/share").split(":")
    os.environ["XDG_DATA_DIRS"] = ":".join([share, *(p for p in existing if p != share)])
    os.environ["PYWEBVIEW_GUI"] = "gtk"
    os.chdir(root)


def display_ready(backend) -> tuple[bool, str]:
    """Check GTK's display before creating a window: importing bindings is insufficient.

    Qt keeps the launcher's environment check because probing its display creates
    a QGuiApplication, which can abort the process instead of reporting failure.
    """
    if str(getattr(backend, "__name__", "")).endswith(".gtk"):
        try:
            from gi.repository import Gdk
        except Exception as exc:  # gi vanished between import and probe
            return False, f"GTK backend without gi.repository.Gdk: {type(exc).__name__}: {exc}"
        if Gdk.Display.get_default() is None:
            return False, "GTK backend has no default display (DISPLAY/WAYLAND_DISPLAY is unusable)"
    return True, ""

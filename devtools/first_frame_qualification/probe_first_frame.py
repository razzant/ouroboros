#!/usr/bin/env python3
"""#1111 native first frame: what a real pywebview 5.4 window shows BEFORE its document, and after.

Qualification equipment for a disposable GitHub-hosted runner, not product. Each case is one window in its own process,
made through the candidate's own seams on the real backend (WinForms + WebView2 on Windows, GTK 3 + WebKitGTK on X11).
A loopback server HOLDS the first response. Frames carry their stage: ``shown`` when pywebview says so, ``loading`` a
bounded crossover around the engine's request, ``held`` while the response is held; the release shows the candidate's
own theme.js and ui.css with a stored Light, Dark or System choice, the page reports what it painted, and ``transition``
and ``painted`` follow it. Samples prove no frame between them. A sample is the whole screen as a PNG (Pillow's
ImageGrab) plus the window's rectangles; judging reads those PNGs back, so a reader sees exactly what was judged.

The controls say what held frames are. Each has a magenta ``background_color`` and a page of one sentinel colour:
``control`` on a url, no seam, runs first on every host; ``control-html`` on the path pre-server windows take, where the
hold is a head script in an already supplied document. On GTK ``control-room`` is ``control`` through the candidate's
own paint, told to use magenta, so the paint's effect shows beside its base. A control whose painted frame lacks the
sentinel, whose hold showed it, or whose frames cannot be read says nothing of its path; ``control``'s stops the matrix.
Losing magenta stops nothing: a renderer may take over from the window's fill (Windows' dark host did, #121212). Held
frames are that path's native fill only where magenta showed while held. Anything else stays a sampled held canvas of
unknown provenance, which supports a light or dark appearance and never an exact fill: only TOLERANCE of a colour on a
native path is a match. ``baseline-*`` cases repeat today's calls, the old colours.

Windows judges the contract exactly on both paths: exit 0 ``mechanical_match_visual_review_pending`` (a reader must
still look), 1 ``failed`` (a candidate case contradicted it under a proven control), 3 ``unknown`` (something was not
observed). GTK judges the same contract where it applies: 0 ``observed_visual_review_pending``, 1, or 3; exact where
``control-room`` showed magenta, by appearance on the html path, whose hold the probe adds. A candidate's url held
frames of the host's opposite appearance fail on either OS; a loading frame of that appearance leaves the run unknown,
and a shown frame only advises. 2 is a refusal. A case past its deadline is killed with its process tree; a close that
never returns ends
by watchdog with exit 4 and says so in teardown.json; renderers left after a case are listed, killed and re-checked, and
a scan that fails is said and stops the matrix. Each run's judgement reaches summary.json and verdict.json before that
scan, so a step killed at its ceiling leaves them: ``unknown``, or ``failed`` if a run already contradicted, never a
match.

  plan | bind --equipment-repo DIR --candidate NEW_DIR --evidence DIR | matrix --source CANDIDATE --out DIR
  run-case --source CANDIDATE --out DIR --case CASE --choice CHOICE | summarize --out DIR --name NAME --outcome OUTCOME
"""
from __future__ import annotations

import argparse
import ast
import itertools
import json
import os
import pathlib
import platform
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import metadata

WINDOWS = sys.platform == "win32"
ROOM = {"light": "#ffffff", "dark": "#0f0f11"}  # web/ui.css --bg-primary of each palette
WHITE, MAGENTA, SENTINEL = "#ffffff", "#ff00ff", "#00a0a0"
TOLERANCE = 2  # per channel; the PNGs keep the raw colours
CAPTION_DOCUMENTED_BUILD = 22000  # DWMWA_USE_IMMERSIVE_DARK_MODE is documented from this build
APP_MODE_KEY = r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize"
TITLE = "ff"  # short, so the caption pixel 60 % across never lands on the title
CASE_SECONDS, CLOSE_SECONDS, EXIT_CLOSE_HUNG = 420, 30, 4
HOST_SETTLE_SECONDS, TAIL_GRACE_SECONDS = 2, 10  # after a host change; for renderers to end
EQUIPMENT_PATHS = ("devtools/first_frame_qualification/", ".github/workflows/first-frame-qualification.yml")
HEX40 = re.compile(r"^[0-9a-f]{40}$")
CONTROLS = {"control": "url", "control-html": "html", "control-room": "url-room"}  # the page path each one speaks for
NATIVE_FILL, HELD_CANVAS, UNPROVEN = "native fill", "held canvas of unknown provenance", "unproven"  # held_provenance
LOADING_FRAMES, LOADING_STEP_SECONDS = 30, 0.1  # at most, between the show and the engine's request
JUDGED_SEAMS = ("main", "setup", "room")  # the candidate's; a baseline only shows what the old calls did
# window: create_window's colour and size (a 1024x768 desktop holds each at 24,24). page: a url the server holds, or
# html= held by a render-blocking head script. seam: the candidate's calls in product order (apply_seam).
CASES = {
    "control": {"window": {"background_color": MAGENTA, "width": 520, "height": 300}, "page": "url", "seam": "none"},
    "control-html": {"window": {"background_color": MAGENTA, "width": 520, "height": 300}, "page": "html",
                     "seam": "none"},
    "control-room": {"window": {"background_color": MAGENTA, "width": 520, "height": 300}, "page": "url",
                     "seam": "room-sentinel"},
    "baseline-main": {"window": {"background_color": "#0f0f11", "width": 960, "height": 640}, "page": "url",
                      "seam": "caption"},
    "baseline-preserver": {"window": {"width": 520, "height": 300}, "page": "html", "seam": "none"},
    "preserver": {"window": {"width": 520, "height": 300}, "page": "html", "seam": "room"},
    "main": {"window": {"background_color": "#0f0f11", "width": 960, "height": 640}, "page": "url", "seam": "main"},
    "setup": {"window": {"width": 900, "height": 640}, "page": "url", "seam": "setup"},
}


def plan_for(windows: bool) -> tuple:
    """(hosts with each one's System palette, runs per host): the smallest set that shows the contract there."""
    if windows:  # Old colours differ from the new: main dark on the light host, a pre-server window white on the dark.
        return ({"light": "light", "dark": "dark"},
                [("control", "system"), ("control-html", "system"), ("baseline-main", "system"),
                 ("baseline-preserver", "system"),
                 ("preserver", "system"), ("main", "light"), ("main", "dark"), ("main", "system"), ("setup", "system")])
    # No baseline: ``control`` is today's url window beside ``control-room``, the candidate's paint. GTK_THEME light and
    # dark, and GTK's own prefer-dark setting (WebKitGTK may read it differently) as a second dark host.
    return ({"light": "light", "dark": "dark", "prefer-dark": "dark"},
            [("control", "system"), ("control-html", "system"), ("control-room", "system"), ("preserver", "system"),
             ("main", "light"), ("main", "dark"), ("main", "system"), ("setup", "system")])


HOSTS, RUNS = plan_for(WINDOWS)
RENDERER = "edgechromium" if WINDOWS else "gtkwebkit2"  # pywebview's own name; anything else proves nothing here
RENDERER_PROCESS = "msedgewebview2.exe" if WINDOWS else "WebKit"
# An image request needs no CORS, so the same beacon leaves a served page and an html= page with no origin.
BEACON = """<script>
addEventListener('load', () => {
    let sent = false;
    const send = () => {
        if (sent) return;
        sent = true;
        const root = document.documentElement;
        new Image().src = 'http://127.0.0.1:%d/painted?d=' + encodeURIComponent(JSON.stringify({
            theme: root.dataset.theme || '', choice: root.dataset.themeChoice || '',
            bg: getComputedStyle(document.body).backgroundColor,
            prefersDark: matchMedia('(prefers-color-scheme: dark)').matches, ua: navigator.userAgent}));
    };
    requestAnimationFrame(() => requestAnimationFrame(send));
    setTimeout(send, 1500);  // an occluded window may never run a frame; the report still says what it holds
});
</script>"""


def rgb_of(colour: str) -> tuple:
    return tuple(int(colour[index:index + 2], 16) for index in (1, 3, 5))


def hex_of(rgb) -> str:
    return "#%02x%02x%02x" % tuple(rgb)[:3]


def close(one: str, other: str) -> bool:
    return all(abs(a - b) <= TOLERANCE for a, b in zip(rgb_of(one), rgb_of(other)))


def dark(colour: str) -> bool:
    red, green, blue = rgb_of(colour)
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue < 128


def read_json(path: pathlib.Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


# ---- the page and the server that holds it ----------------------------------------------------------------------

def page_html(source: pathlib.Path, port: int, case: str, choice: str) -> str:
    """The url control's one-colour page, else the candidate's own standalone page (theme.js and ui.css inlined)."""
    if case in ("control", "control-room"):  # no theme: its paint cannot be mistaken for any fill or room
        return (f'<!DOCTYPE html><html><head><meta charset="utf-8"><title>{TITLE}</title></head>'
                f'<body style="margin:0;min-height:100vh;background:{SENTINEL}">{BEACON % port}</body></html>')
    from ouroboros.standalone_page import themed_page

    if CASES[case]["page"] == "html":  # launcher._window_page's document, plus the head script that holds it
        css = next(ast.literal_eval(node.value) for node in ast.parse((source / "launcher.py").read_text("utf-8")).body
                   if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "_WINDOW_CSS" for t in node.targets))
        if case == "control-html":  # the preserver's own document, held alike: only its release paints the sentinel
            css += f"html, body {{ background: {SENTINEL} !important; }}"
        body = ('<div class="window"><h2>Ouroboros is already running</h2>'
                '<p>Ouroboros is already running from another copy. Quit it, then open this copy.</p></div>')
        return themed_page(source / "web", TITLE, body + BEACON % port, css,
                           f'<script src="http://127.0.0.1:{port}/held.js"></script>\n')
    store = "" if choice == "system" else f"localStorage.setItem('ouroboros.theme', {json.dumps(choice)});"
    head = f"<script>try {{ localStorage.removeItem('ouroboros.theme'); {store} }} catch (_) {{}}</script>\n"
    body = f'<p style="margin: 24px 32px; font-size: 14px;">first-frame probe: {case}, stored {choice}</p>'
    css = ("body { margin: 0; min-height: 100vh; background: var(--bg-primary, Canvas); "
           "color: var(--text-primary, CanvasText); font-family: var(--font-sans, system-ui); }")
    return themed_page(source / "web", TITLE, body + BEACON % port, css, head)


class Held:
    """A loopback server: ``/page`` and ``/held.js`` wait for ``release``; ``/painted`` takes the page's report."""

    def __init__(self) -> None:
        self.release, self.requested, self.painted = threading.Event(), threading.Event(), threading.Event()
        self.report, self.requests, self.html = {}, [], ""
        held = self

        class Handler(BaseHTTPRequestHandler):
            def answer(self, status: int, kind: str = "", body: bytes = b"") -> None:
                self.send_response(status)
                self.send_header("Cache-Control", "no-store")
                if kind:
                    self.send_header("Content-Type", kind)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:
                parts = urllib.parse.urlsplit(self.path)
                if parts.path in ("/page", "/held.js"):
                    held.requests.append(parts.path)
                    held.requested.set()
                    held.release.wait(300)
                    page = parts.path == "/page"
                    self.answer(200, "text/html; charset=utf-8" if page else "text/javascript",
                                held.html.encode() if page else b"")
                elif parts.path == "/painted":
                    for value in urllib.parse.parse_qs(parts.query).get("d", []):
                        held.report.update(json.loads(value))
                        held.painted.set()
                    self.answer(204)
                else:
                    self.answer(404)

            def log_message(self, *args) -> None:
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_port
        self.url = f"http://127.0.0.1:{self.port}/page"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()


# ---- what the screen shows, and where the window is on it -------------------------------------------------------

def grab(path: pathlib.Path) -> None:
    """The whole primary screen as the display composes it (Windows: BitBlt with CAPTUREBLT; X11: the root window)."""
    from PIL import ImageGrab

    image = ImageGrab.grab(include_layered_windows=True) if WINDOWS else ImageGrab.grab(xdisplay=os.environ["DISPLAY"])
    image.save(path)


def _win32():
    import ctypes
    from ctypes import wintypes as w

    user32, dwm = ctypes.WinDLL("user32"), ctypes.WinDLL("dwmapi")  # private handles: argtypes stay ours
    for function, result, arguments in (
            (user32.GetWindowRect, w.BOOL, (w.HWND, ctypes.POINTER(w.RECT))),
            (user32.GetClientRect, w.BOOL, (w.HWND, ctypes.POINTER(w.RECT))),
            (user32.ClientToScreen, w.BOOL, (w.HWND, ctypes.POINTER(w.POINT))),
            (dwm.DwmGetWindowAttribute, ctypes.c_long, (w.HWND, w.DWORD, ctypes.c_void_p, w.DWORD))):
        function.restype, function.argtypes = result, arguments
    return ctypes, w, user32, dwm


def caption_attribute(hwnd: int):
    """DWMWA_USE_IMMERSIVE_DARK_MODE as DWM reads it back; the HRESULT as text where this build will not."""
    ctypes, w, _user32, dwm = _win32()
    value = w.BOOL()
    result = dwm.DwmGetWindowAttribute(w.HWND(hwnd), 20, ctypes.byref(value), ctypes.sizeof(value))
    return bool(value.value) if result >= 0 else f"hresult {result & 0xFFFFFFFF:#010x}"


def on_gtk_thread(read):
    """``read()`` on the GTK main loop, which owns every widget; its exception is raised here."""
    from gi.repository import GLib

    found, ready = {}, threading.Event()

    def call() -> bool:
        try:
            found["value"] = read()
        except Exception as exc:
            found["error"] = exc
        ready.set()
        return False

    GLib.idle_add(call)
    if not ready.wait(30):
        raise RuntimeError("the GTK main loop did not answer within 30 s")
    if "error" in found:
        raise found["error"]
    return found["value"]


def window_rects(window, hwnd: int) -> dict:
    """The window's visible ``frame`` and its ``client`` area on the screen, each [left, top, right, bottom]."""
    if WINDOWS:
        ctypes, w, user32, dwm = _win32()
        frame, client, origin = w.RECT(), w.RECT(), w.POINT(0, 0)
        if dwm.DwmGetWindowAttribute(w.HWND(hwnd), 9, ctypes.byref(frame), ctypes.sizeof(frame)) < 0:  # extended bounds
            user32.GetWindowRect(w.HWND(hwnd), ctypes.byref(frame))
        user32.GetClientRect(w.HWND(hwnd), ctypes.byref(client))
        user32.ClientToScreen(w.HWND(hwnd), ctypes.byref(origin))
        return {"frame": [frame.left, frame.top, frame.right, frame.bottom],
                "client": [origin.x, origin.y, origin.x + client.right, origin.y + client.bottom],
                "caption_attribute": caption_attribute(hwnd)}

    def read() -> dict:
        gdk = window.native.get_window()
        *_, x, y = gdk.get_origin()  # (result, x, y) in PyGObject
        frame = gdk.get_frame_extents()
        return {"frame": [frame.x, frame.y, frame.x + frame.width, frame.y + frame.height],
                "client": [x, y, x + gdk.get_width(), y + gdk.get_height()]}

    return on_gtk_thread(read)


def toolkit_facts(window) -> dict:
    """GTK's theme as GTK itself resolved it, the web view's own colour and opacity as WebKit answers them, and the
    toolkit versions; diagnostic, never the verdict."""
    def channels(colour) -> str:
        return hex_of(round(part * 255) for part in (colour.red, colour.green, colour.blue))

    def read() -> dict:
        settings = window.native.get_settings()
        known, colour = window.native.get_style_context().lookup_color("theme_bg_color")
        gtk, view = window.gui.gtk, window.gui.BrowserView.instances[window.uid].webview
        return {"theme": settings.get_property("gtk-theme-name"), "GTK_THEME": os.environ.get("GTK_THEME"),
                "prefer_dark": bool(settings.get_property("gtk-application-prefer-dark-theme")),
                "theme_bg_color": channels(colour) if known else None,
                "web_view_background": channels(view.get_background_color()), "web_view_opacity": view.get_opacity(),
                "webkit2": list(getattr(window.gui, "webkit_ver", ())),
                "gtk": [gtk.get_major_version(), gtk.get_minor_version(), gtk.get_micro_version()]}

    try:
        return {} if WINDOWS else on_gtk_thread(read)
    except Exception as exc:
        return {"unread": f"{type(exc).__name__}: {exc}"}


def read_app_mode() -> str:
    """AppsUseLightTheme as read now, for the record ("absent" when there is none)."""
    if not WINDOWS:
        return ""
    import winreg

    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, APP_MODE_KEY) as key:
            return repr(winreg.QueryValueEx(key, "AppsUseLightTheme"))
    except OSError:
        return "absent"


def set_app_mode(value):
    """Set AppsUseLightTheme (a DWORD), or delete it for ``None``; the previous (value, type), None when absent."""
    import winreg

    with winreg.CreateKeyEx(winreg.HKEY_CURRENT_USER, APP_MODE_KEY, 0, winreg.KEY_READ | winreg.KEY_SET_VALUE) as key:
        try:
            before = winreg.QueryValueEx(key, "AppsUseLightTheme")
        except OSError:
            before = None
        if value is not None:
            winreg.SetValueEx(key, "AppsUseLightTheme", 0, winreg.REG_DWORD, value)
        elif before is not None:
            winreg.DeleteValue(key, "AppsUseLightTheme")
    return before


# ---- one window -----------------------------------------------------------------------------------------------------

def apply_seam(seam: str, window, appearance, launcher_window_chrome) -> None:
    """The candidate's calls, in product order. ``caption`` is today's main window: the caption's owner, no room."""
    if seam == "main":  # launcher_background.Background.attach's order
        appearance.attach(window)
        launcher_window_chrome.WindowChrome().attach(window)
    elif seam == "setup":  # launcher_onboarding's order
        appearance.attach(window)
        launcher_window_chrome.paint_room(window)
    elif seam == "room":  # launcher.py's three windows shown before the server
        launcher_window_chrome.paint_room(window)
    elif seam == "caption":
        appearance.attach(window)
    elif seam == "room-sentinel":  # the candidate's whole GTK path, its room a colour no page or theme has
        launcher_window_chrome.ROOM = dict.fromkeys(launcher_window_chrome.ROOM, MAGENTA)
        launcher_window_chrome.paint_room(window)


def run_case(args, webview=None) -> int:
    """One window: samples from its show, while held, through the release, and painted; case.json, then teardown.json."""
    out, source = pathlib.Path(args.out).resolve(), pathlib.Path(args.source).resolve()
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(source))
    if webview is None:
        import webview
    from ouroboros import launcher_appearance, launcher_window_chrome

    spec = CASES[args.case]
    held = Held()
    held.html = page_html(source, held.port, args.case, args.choice)
    record = {"case": args.case, "choice": args.choice, "spec": spec, "platform": platform.platform(),
              "python": sys.version, "packages": {name: metadata.version(name) for name in ("pywebview", "pillow")},
              "windows_build": sys.getwindowsversion().build if WINDOWS else None, "app_mode": read_app_mode(),
              "host_palette": launcher_appearance.host_palette(), "samples": [], "errors": []}
    appearance = launcher_appearance.NativeAppearance()
    bridged = spec["seam"] in ("main", "setup", "caption")

    class Api:  # the bridge theme.js calls, as launcher.MainApi and OnboardingHostApi carry it
        def set_native_appearance(self, theme="", page=None, sequence=None):
            return appearance.request(theme, page, sequence)

    target = {"html": held.html} if spec["page"] == "html" else {"url": held.url}
    window = webview.create_window(f"{TITLE} {args.case}", js_api=Api() if bridged else None, x=24, y=24, **target,
                                   **spec["window"])
    apply_seam(spec["seam"], window, appearance, launcher_window_chrome)
    record["background_color_after_seams"] = window.background_color
    handle = {"hwnd": 0}
    if WINDOWS:  # before_show runs on the form's UI thread, where its Handle may be read
        window.events.before_show += lambda: handle.update(hwnd=int(window.native.Handle.ToInt64()))
    counter, started, finished, destroyed = itertools.count(), time.monotonic(), threading.Event(), [0.0]

    def teardown(closed: bool, note: str) -> None:
        after = round(time.monotonic() - destroyed[0], 1) if destroyed[0] else None
        (out / "teardown.json").write_text(json.dumps({"closed": closed, "note": note, "seconds_after_destroy": after}),
                                           encoding="utf-8")

    def watchdog() -> None:  # a close that never returns is its own outcome, never a clean exit
        if not finished.wait(CLOSE_SECONDS):
            teardown(False, f"webview.start had not returned {CLOSE_SECONDS} s after destroy(); exit {EXIT_CLOSE_HUNG}")
            os._exit(EXIT_CLOSE_HUNG)

    def shot(label: str, stage: str) -> None:
        name = f"{next(counter):02d}-{label}.png"
        sample = {"label": label, "stage": stage, "png": name, "requests": len(held.requests),
                  "t": round(time.monotonic() - started, 3), **window_rects(window, handle["hwnd"])}
        grab(out / name)
        record["samples"].append(sample)

    def drive() -> None:
        try:
            if not window.events.shown.wait(180):
                raise RuntimeError("the window was never shown")
            record["renderer"] = str(getattr(window.gui, "renderer", ""))
            record["toolkit"] = toolkit_facts(window)
            shot("shown", "shown")  # kept, not judged: it may precede the first paint
            for index in range(LOADING_FRAMES):  # bounded: a crossover long enough to be sampled shows here
                if held.requested.wait(LOADING_STEP_SECONDS):
                    break
                shot(f"loading-{index:02d}", "loading")
            if not held.requested.wait(180):
                raise RuntimeError("the engine never asked for the held response, so no frame here is pre-document")
            for index in range(2):  # the surface may change as the load begins, before the first held frame
                shot(f"loading-requested-{index}", "loading")
                time.sleep(LOADING_STEP_SECONDS)
            for index, delay in enumerate((0.3, 0.7, 1.0)):  # the response is held: each frame precedes the page
                time.sleep(delay)
                shot(f"held-{index}", "held")
            record["toolkit_held"] = toolkit_facts(window)
            held.release.set()
            for index in range(12):
                shot(f"transition-{index:02d}", "transition")
                if held.painted.is_set():
                    break
                time.sleep(0.05)
            if not held.painted.wait(120):
                raise RuntimeError("the released page never reported its paint")
            record["page"] = dict(held.report)
            if WINDOWS and bridged:  # theme.js reports over the bridge once pywebview injected it: bounded wait
                for _ in range(60):
                    seen = caption_attribute(handle["hwnd"])
                    if not isinstance(seen, bool) or seen == (record["page"].get("theme") == "dark"):
                        break
                    time.sleep(0.2)
            time.sleep(0.8)
            shot("painted", "painted")
        except Exception as exc:  # recorded, never a pass
            record["errors"].append(f"{type(exc).__name__}: {exc}")
        finally:
            held.release.set()
            (out / "case.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
            destroyed[0] = time.monotonic()
            threading.Thread(target=watchdog, daemon=True).start()
            window.destroy()

    try:  # a fresh WebView profile per case
        webview.start(func=drive, private_mode=False, storage_path=tempfile.mkdtemp(prefix="ff-profile-"))
    finally:
        finished.set()
        held.server.shutdown()
    teardown(True, "webview.start returned")
    return 3 if record["errors"] else 0


# ---- judging --------------------------------------------------------------------------------------------------------

def colours(png: pathlib.Path, rect: list, whole: bool) -> list:
    """Colours at three margin points of ``rect`` (no page draws within 6 px of them) and, when ``whole``, an 8x8 grid."""
    from PIL import Image

    with Image.open(png) as image:
        rgb = image.convert("RGB")
    left, top, right, bottom = rect
    width, height = right - left, bottom - top
    points = [(6, height - 6), (6, 6), (width // 2, height - 6)]
    if whole:
        points += [(width * column // 8 + width // 16, height * row // 8 + height // 16)
                   for column in range(8) for row in range(8)]
    if not (0 <= left and 0 <= top and right <= rgb.width and bottom <= rgb.height and width > 12 and height > 12):
        raise ValueError(f"the window's client {rect} is not inside the captured screen {rgb.size}")
    return sorted({hex_of(rgb.getpixel((left + x, top + y))) for x, y in points})


def caption_pixel(png: pathlib.Path, frame: list, client: list):
    """The caption band's colour 60 % across (past the short title, before the buttons); None without a band."""
    from PIL import Image

    band = client[1] - frame[1]
    if band < 8:
        return None
    with Image.open(png) as image:
        return hex_of(image.convert("RGB").getpixel((frame[0] + (frame[2] - frame[0]) * 6 // 10, frame[1] + band // 2)))


def expected(case: str, choice: str, host: str) -> dict:
    """The contract for one run: its room before the page, its page's after. A caption nobody tints is the light default
    (captions are judged on Windows alone)."""
    palette = HOSTS[host]
    page = palette if choice == "system" else choice
    seam = CASES[case]["seam"]
    fill = {"main": ROOM[palette], "setup": ROOM[palette], "preserver": ROOM[palette], "control": MAGENTA,
            "control-html": MAGENTA, "control-room": MAGENTA, "baseline-main": ROOM["dark"],
            "baseline-preserver": WHITE}[case]
    bridged = seam in ("main", "setup", "caption")
    held_caption = dark(fill) if bridged or seam == "room" else False
    return {"palette": palette, "page_palette": page, "pre_document": fill,
            "painted": SENTINEL if case in CONTROLS else ROOM[page],
            "caption_held": held_caption, "caption_painted": page == "dark" if bridged else held_caption}


def judge_run(case: str, choice: str, host: str, record: dict, facts: dict, case_dir: pathlib.Path) -> dict:
    """``invalid``: why these frames cannot speak (a control's gap reads only this). ``unobserved`` adds the process
    and teardown facts. ``contradictions``: candidate cases against the contract; ``held_contradictions``: those that
    read frames as the window's fill, which their page path's control must support; ``opposite_appearance``: held frames
    of the host's opposite light or dark, which a canvas can show as well as a fill; a baseline that misbehaves leaves
    the old-colour comparison unproven. Controls are judged in control_gap and held_provenance; judge_matrix routes all
    of this by path, and adds the renderer tail, which is scanned after this."""
    invalid, process, contradictions, held_contradictions, advisories = [], [], [], [], []
    exit_code = facts.get("exit")
    if exit_code != 0:
        process.append(f"case process exit {exit_code}" + (": its window's close hung" if exit_code == EXIT_CLOSE_HUNG
                                                            else ""))
    teardown = facts.get("teardown")
    if teardown is None:
        process.append("no teardown record: the window's close was not observed")
    elif not teardown.get("closed"):
        process.append(f"window close hung: {teardown.get('note')}")
    if not record:
        invalid.append("no case record")
    invalid += [f"error: {error}" for error in record.get("errors", [])]
    renderer_ok = record.get("renderer") == RENDERER
    if record and not renderer_ok:
        invalid.append(f"renderer {record.get('renderer')!r} is not {RENDERER}")
    samples = record.get("samples", [])
    shown, loading, held, painted, transition = ([sample for sample in samples if sample["stage"] == stage]
                                                 for stage in ("shown", "loading", "held", "painted", "transition"))
    if len(held) != 3 or not all(sample.get("requests") for sample in held):
        invalid.append(f"{len(held)} held frames, requests {[sample.get('requests') for sample in held]}")
    if len(painted) != 1:
        invalid.append(f"{len(painted)} painted frames")
    for sample in samples:
        png = case_dir / sample["png"]
        try:
            sample["colours"] = colours(png, sample["client"], whole=sample["stage"] in ("shown", "loading", "held"))
            if WINDOWS:
                sample["caption_pixel"] = caption_pixel(png, sample["frame"], sample["client"])
        except Exception as exc:  # a missing, empty or unreadable image is said, never skipped
            invalid.append(f"{sample['label']}: {sample['png']} unread: {type(exc).__name__}: {exc}")

    def seen(group: list) -> list:
        return sorted({colour for sample in group for colour in sample.get("colours", [])})

    first, crossing, before, after, between = seen(shown), seen(loading), seen(held), seen(painted), seen(transition)
    page = record.get("page") or {}
    if record and not page:
        invalid.append("the page reported nothing")
    entry = {"renderer_ok": renderer_ok, "invalid": invalid, "unobserved": invalid + process, "advisories": advisories,
             "contradictions": contradictions, "held_contradictions": held_contradictions, "opposite_appearance": [],
             "shown_seen": first, "loading_seen": crossing, "pre_document_seen": before, "painted_seen": after,
             "transition_seen": between, "page_reported": page, "host_palette": record.get("host_palette"),
             "background_color_after_seams": record.get("background_color_after_seams"),
             "app_mode": record.get("app_mode"), "toolkit": record.get("toolkit"),
             "toolkit_held": record.get("toolkit_held"),
             "timeline": [{key: sample.get(key) for key in ("label", "t", "requests", "colours")}
                          for sample in samples]}
    if case in CONTROLS:
        return entry
    want = entry["expected"] = expected(case, choice, host)
    judged = CASES[case]["seam"] in JUDGED_SEAMS
    host_dark = want["palette"] == "dark"

    def opposite(group: list) -> list:
        return [colour for colour in group if dark(colour) != host_dark]

    if judged and opposite(before):
        entry["opposite_appearance"].append(f"held frames show {opposite(before)} on a {want['palette']} host")
    if judged and opposite(first):  # it may precede the first paint, so it only advises
        advisories.append(f"shown frames show {opposite(first)} on a {want['palette']} host")
    if judged and opposite(crossing):  # seen, so never a match; a sample in a crossover is not enough to fail
        entry["unobserved"].append(f"loading frames show {opposite(crossing)} on a {want['palette']} host")

    def against(text: str, kept=contradictions) -> None:
        (kept if judged else entry["unobserved"]).append(text if judged else f"old-colour comparison unproven: {text}")

    if before and not all(close(colour, want["pre_document"]) for colour in before):
        against(f"pre-document frames show {before}, expected {want['pre_document']}", held_contradictions)
    if after and not all(close(colour, want["painted"]) for colour in after):
        against(f"painted frame shows {after}, expected {want['painted']}")
    extra = [colour for colour in between
             if not any(close(colour, known) for known in (want["pre_document"], want["painted"]))]
    if extra:
        (held_contradictions if judged else advisories).append(
            f"page transition showed {extra} beside {want['pre_document']} and {want['painted']}")
    if page and page.get("theme") != want["page_palette"]:
        against(f"page painted {page.get('theme')!r}, expected {want['page_palette']}")
    if page and page.get("prefersDark") != (want["palette"] == "dark"):
        against(f"page prefers-color-scheme dark={page.get('prefersDark')} on a {want['palette']} host")
    if not WINDOWS:  # GTK's palette is read from its theme's colour in before_show, and its caption is the theme's
        return entry
    if record and record.get("host_palette") != want["palette"]:
        against(f"host_palette read {record.get('host_palette')!r} on a {want['palette']} host")
    documented = (record.get("windows_build") or 0) >= CAPTION_DOCUMENTED_BUILD
    for sample, want_dark in ([(held[-1], want["caption_held"])] if held else []) + (
            [(painted[-1], want["caption_painted"])] if painted else []):
        attribute, pixel = sample.get("caption_attribute"), sample.get("caption_pixel")
        problems = [f"attribute {attribute}"] if isinstance(attribute, bool) and attribute != want_dark else []
        if pixel is None:
            entry["unobserved"].append(f"{sample['label']}: no caption pixel")
        elif dark(pixel) != want_dark:
            problems.append(f"pixel {pixel}")
        if problems:
            message = f"{sample['label']}: caption dark={want_dark} expected, {', '.join(problems)}"
            if documented:
                against(message)
            else:
                advisories.append(message)
    if record and not documented:
        advisories.append(f"build {record.get('windows_build')} predates the documented caption attribute")
    return entry


def control_gap(entry) -> str:
    """Why a control says nothing of its page path: not run, unreadable, page unseen, hold not holding; else ""."""
    if entry is None:
        return "no control run"
    if entry["invalid"]:
        return f"its frames cannot speak: {entry['invalid']}"
    held, painted = entry["pre_document_seen"], entry["painted_seen"]
    if not painted or not all(close(colour, SENTINEL) for colour in painted):
        return f"the painted frame shows {painted}, not the control page's {SENTINEL}: the capture does not see the page"
    if any(close(colour, SENTINEL) for colour in held + entry["loading_seen"]):
        return (f"frames before the release show the page's {SENTINEL} ({held + entry['loading_seen']}): "
                "they are not pre-document")
    return ""


def held_provenance(entry) -> str:
    """What frames held on a control's page path are: its magenta seen while held, NATIVE_FILL; any other colour, a
    sampled HELD_CANVAS nothing here attributes (the shown frames say whether the window's fill was seen earlier); no
    proof, UNPROVEN. Samples, not every frame."""
    gap = control_gap(entry)
    if gap:
        return f"{UNPROVEN}: {gap}"
    held = entry["pre_document_seen"]
    if all(close(colour, MAGENTA) for colour in held):
        return NATIVE_FILL
    return f"{HELD_CANVAS}: {held}; shown {entry['shown_seen']}"


def control_for(case: str) -> str:
    """The control whose held frames say what this case's are: its page path's, and on GTK its own paint's."""
    if CASES[case]["page"] == "html":
        return "control-html"
    return "control-room" if not WINDOWS and CASES[case]["seam"] in ("main", "setup") else "control"


def judge_matrix(summary: dict) -> dict:
    runs = summary.get("runs", [])

    def control(host: str, case: str):
        return next((run for run in runs if run["host"] == host and run["case"] == case), None)

    proof = {host: control_gap(control(host, "control")) for host in HOSTS}
    provenance = {host: {CONTROLS[case]: held_provenance(control(host, case)) for case, _ in RUNS if case in CONTROLS}
                  for host in HOSTS}

    def exact_on(path: str) -> bool:  # GTK's html path shows its toolkit's colour while the probe holds it
        return WINDOWS or path == "url-room"

    contradictions, unobserved, advisories, appearance = [], [], [], []
    for run in runs:
        where = f"{run['host']}/{run['case']}-{run['choice']}"
        unobserved += [f"{where}: {reason}" for reason in run["unobserved"] + tail_reasons(run["tail"])]
        advisories += [f"{where}: {reason}" for reason in run["advisories"]]
        if run["case"] in CONTROLS:
            continue
        named = CONTROLS[control_for(run["case"])]
        path = provenance[run["host"]][named]
        speaks = run["renderer_ok"] and not proof[run["host"]] and not path.startswith(UNPROVEN)
        url = CASES[run["case"]]["page"] == "url"
        for texts, proven, doubt, kept in (
                (run["contradictions"], speaks, "on an unproven capture or renderer", unobserved),
                (run["held_contradictions"], speaks and path == NATIVE_FILL,
                 "held frames no control showed to be the window's fill",
                 unobserved if exact_on(named) or not speaks else advisories),
                # A url response is really waited for, so its held frames are what a person sees. The html hold is the
                # probe's own: the same colours there are said, and only its native fill can fail.
                (run["opposite_appearance"], speaks and (url or path == NATIVE_FILL),
                 "on a hold the probe adds, or an unproven path", unobserved)):
            for text in texts:
                if proven:
                    contradictions.append(f"{where}: {text}")
                else:
                    kept.append(f"{where}: {doubt}: {text}")
        if (CASES[run["case"]]["seam"] in JUDGED_SEAMS and speaks and path != NATIVE_FILL
                and run["pre_document_seen"] and not run["opposite_appearance"]):
            appearance.append(f"{where}: held {run['pre_document_seen']} is the {HOSTS[run['host']]} host's appearance "
                              "on a canvas: observed, and no exact fill is claimed")
    planned = len(HOSTS) * len(RUNS)
    if len(runs) != planned:
        unobserved.append(f"{len(runs)} of {planned} runs recorded: {summary.get('stopped') or 'the matrix ended early'}")
    unobserved += [f"{host}: the control does not prove the capture: {reason}" for host, reason in proof.items() if reason]
    unobserved += [f"{host}: frames held on the {path} path are {text}: they say nothing of the window's fill"
                   for host, paths in provenance.items() for path, text in paths.items()
                   if text.startswith(UNPROVEN) or (exact_on(path) and text != NATIVE_FILL)]
    if WINDOWS and "app_mode_original" in summary and summary.get("app_mode_restored") is not True:
        unobserved.append(f"the host's app mode was not restored: {summary.get('app_mode_after_restore')!r}")
    matched = "mechanical_match_visual_review_pending" if WINDOWS else "observed_visual_review_pending"
    status = "failed" if contradictions else "unknown" if unobserved else matched
    observations = [  # what each run showed, stage by stage; what it means is judged above
        f"{run['host']}/{run['case']}-{run['choice']}: shown {run['shown_seen']} loading {run['loading_seen']} "
        f"held {run['pre_document_seen']} -> painted {run['painted_seen']}; "
        f"window colour {run.get('background_color_after_seams')}; page {run['page_reported'].get('theme')!r} "
        f"prefersDark={run['page_reported'].get('prefersDark')}; toolkit {run.get('toolkit')}" for run in runs]
    return {"status": status, "contradictions": contradictions, "unobserved": unobserved, "advisories": advisories,
            "appearance": appearance, "capture_proof": {host: reason or "proven" for host, reason in proof.items()},
            "held_provenance": provenance, "observations": observations,
            "scope": "component frames of real windows through the candidate's seams, sampled at the show, up to "
                     f"{LOADING_FRAMES} times until the held request, twice right after it, then while held: not every "
                     "early flash or compositor frame, High Contrast pixels, packaged cold start, source-launcher "
                     "entry, Qt, Wayland or the system WKWebView; the runner's disposal is GitHub's"}


EXIT_OF_STATUS = {"mechanical_match_visual_review_pending": 0, "observed_visual_review_pending": 0, "failed": 1,
                  "unknown": 3}


# ---- the matrix ---------------------------------------------------------------------------------------------------

def list_renderers() -> list:
    """Renderer processes alive now, as "pid name": WebView2's on Windows, WebKitGTK's elsewhere."""
    if WINDOWS:
        listing = subprocess.run(["tasklist", "/FI", f"IMAGENAME eq {RENDERER_PROCESS}", "/FO", "CSV", "/NH"],
                                 capture_output=True, text=True, check=True, timeout=30).stdout
        return [f"{line.split(',')[1].strip(chr(34))} {RENDERER_PROCESS}" for line in listing.splitlines()
                if line.lower().startswith(f'"{RENDERER_PROCESS}"')]
    result = subprocess.run(["pgrep", "-l", RENDERER_PROCESS], capture_output=True, text=True, timeout=30)
    if result.returncode != 1:  # pgrep's documented no-match exit, not a failed observation
        result.check_returncode()
    return [line for line in result.stdout.splitlines() if line]


def kill_process(pid: str) -> None:
    subprocess.run(["taskkill", "/F", "/PID", pid] if WINDOWS else ["kill", "-KILL", pid], capture_output=True)


def scan_failure(exc: Exception) -> str:
    stderr = str(getattr(exc, "stderr", None) or "").strip()  # what tasklist or pgrep said; str(exc) omits it
    return f"{type(exc).__name__}: {exc}" + (f"; stderr: {stderr}" if stderr else "")


def renderer_tail(before: set, renderers=list_renderers, kill=kill_process) -> dict:
    """Renderers a finished case left (none alive before the matrix counts), after a grace; killed, re-checked.
    A scan or kill that fails is ``scan_error``, and what was not read stays None: never "none", never a clean kill."""
    alive, left = None, None
    try:
        deadline = time.monotonic() + TAIL_GRACE_SECONDS
        while (alive := [line for line in renderers() if line not in before]) and time.monotonic() < deadline:
            time.sleep(0.5)
        for line in alive:
            kill(line.split()[0])
        if alive:
            time.sleep(1)
        left = [line for line in renderers() if line in alive] if alive else []
    except Exception as exc:
        traceback.print_exc()  # the matrix's log keeps the trace
        return {"alive": alive, "left": left, "scan_error": scan_failure(exc)}
    return {"alive": alive, "left": left}


def tail_reasons(tail) -> list:
    """What a case's renderer scan adds to its unobserved facts; None is a scan that never ended."""
    if tail is None:
        return ["the renderer scan after this case had not finished"]
    reasons = []
    if tail["alive"]:
        left = "unread" if tail["left"] is None else tail["left"]
        reasons.append(f"renderer processes outlived the case: {tail['alive']}; alive after kill: {left}")
    if tail.get("scan_error"):
        reasons.append(f"renderer scan failed, so what this case left and its cleanup are unknown: "
                       f"{tail['scan_error']}")
    return reasons


def launch_case(command: list, env: dict, log_path: pathlib.Path):
    """Run one case in its own process tree: its exit code, or a text saying the tree was killed at the deadline."""
    with open(log_path, "wb") as log:
        kwargs = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if WINDOWS else {"start_new_session": True}
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, **kwargs)
        try:
            return process.wait(CASE_SECONDS)
        except subprocess.TimeoutExpired:
            if WINDOWS:
                subprocess.run(["taskkill", "/T", "/F", "/PID", str(process.pid)], capture_output=True)
            else:
                os.killpg(process.pid, signal.SIGKILL)
            try:
                process.wait(30)
            except subprocess.TimeoutExpired:
                return f"timeout after {CASE_SECONDS} s; the killed process did not exit"
            return f"timeout after {CASE_SECONDS} s; its process tree was killed"


def case_environment(host: str, out: pathlib.Path) -> dict:
    env = dict(os.environ)
    if WINDOWS:
        return env
    roots = {name: out / "scratch" / f"xdg-{name}-{host}" for name in ("config", "data", "cache")}
    for root in roots.values():
        root.mkdir(parents=True, exist_ok=True)
    env.update(XDG_CONFIG_HOME=str(roots["config"]), XDG_DATA_HOME=str(roots["data"]), XDG_CACHE_HOME=str(roots["cache"]),
               PYWEBVIEW_GUI="gtk", GDK_BACKEND="x11", NO_AT_BRIDGE="1")
    env.pop("GTK_THEME", None)
    if host == "prefer-dark":  # GTK's own dark preference, with no GTK_THEME to answer first
        (roots["config"] / "gtk-3.0").mkdir(exist_ok=True)
        (roots["config"] / "gtk-3.0" / "settings.ini").write_text(
            "[Settings]\ngtk-theme-name=Adwaita\ngtk-application-prefer-dark-theme=1\n", encoding="utf-8")
    else:
        env["GTK_THEME"] = "Adwaita:dark" if host == "dark" else "Adwaita"
    return env


def write_summary(out: pathlib.Path, summary: dict) -> dict:
    """summary.json and verdict.json as they stand. judge_matrix counts the runs, so a partial matrix reads ``unknown``,
    or ``failed`` once a run contradicted: what a killed step leaves is what its report says."""
    verdict = summary["verdict"] = judge_matrix(summary)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / "verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    return verdict


def matrix(args, launch=launch_case, renderers=list_renderers, kill=kill_process) -> int:
    out = pathlib.Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    summary: dict = {"platform": platform.platform(), "hosts": HOSTS, "runs_per_host": RUNS, "runs": [],
                     "renderers_before": None}  # None: unread, which no empty list may stand for
    before = None
    try:
        before = set(renderers())
        summary["renderers_before"] = sorted(before)
    except Exception as exc:  # no case's renderers could be told from these, so no case runs
        traceback.print_exc()
        summary["stopped"] = f"no case ran: the renderers alive before the matrix were not read: {scan_failure(exc)}"
        print(summary["stopped"], flush=True)
    original, changed = None, False
    try:
        for host in HOSTS if before is not None else ():
            if WINDOWS:
                previous = set_app_mode(1 if host == "light" else 0)
                if not changed:
                    original, changed = previous, True
                    summary["app_mode_original"] = repr(previous)
                summary.setdefault("app_mode_set", {})[host] = read_app_mode()
                time.sleep(HOST_SETTLE_SECONDS)
            env = case_environment(host, out)
            for case, choice in RUNS:
                case_dir = out / host / f"{case}-{choice}"
                case_dir.mkdir(parents=True, exist_ok=True)
                command = [sys.executable, "-I", "-B", os.path.abspath(__file__), "run-case", "--source", args.source,
                           "--out", str(case_dir), "--case", case, "--choice", choice]
                started = time.monotonic()
                facts = {"exit": launch(command, env, case_dir / "process.log")}
                facts.update(seconds=round(time.monotonic() - started, 1),
                             teardown=read_json(case_dir / "teardown.json"))
                entry = {"host": host, "case": case, "choice": choice, "tail": None, **facts}
                entry.update(judge_run(case, choice, host, read_json(case_dir / "case.json") or {}, facts, case_dir))
                summary["runs"].append(entry)
                write_summary(out, summary)  # this run's outcome is kept before the scan can fail or the step end
                tail = entry["tail"] = renderer_tail(before, renderers, kill)
                write_summary(out, summary)  # and what it found, for a step that ends during the next case
                print(json.dumps({key: entry[key] for key in ("host", "case", "choice", "exit", "seconds", "tail",
                                                              "shown_seen", "loading_seen", "pre_document_seen",
                                                              "painted_seen", "unobserved", "contradictions",
                                                              "held_contradictions", "opposite_appearance")}),
                      flush=True)
                if case == "control" and control_gap(entry):  # capture, hold or provisioning failed: nothing else reads
                    summary["stopped"] = f"stopped after {host}'s control: {control_gap(entry)}"
                elif tail.get("scan_error"):  # the next case could not tell its own renderers from these
                    summary["stopped"] = (f"stopped after {host}/{case}-{choice}: renderers not read: "
                                          f"{tail['scan_error']}")
                if summary.get("stopped"):
                    print(summary["stopped"], flush=True)
                    break
            if summary.get("stopped"):
                break
    finally:
        if WINDOWS and changed:
            set_app_mode(original[0] if original else None)
            after = read_app_mode()
            summary["app_mode_after_restore"] = after
            summary["app_mode_restored"] = after == ("absent" if original is None else repr(original))
        verdict = write_summary(out, summary)  # also while an exception passes: its trace goes on to the log
    print(json.dumps(verdict, indent=2), flush=True)
    return EXIT_OF_STATUS[verdict["status"]]


# ---- identity and report --------------------------------------------------------------------------------------------

def bind(args) -> int:
    """Two identities apart: the candidate alone at its bound commit and tree, and the one equipment commit above it."""
    inputs = json.loads((pathlib.Path(__file__).resolve().parent / "inputs.json").read_text(encoding="utf-8"))
    sha, tree = str(inputs["candidate"]["sha"]), str(inputs["candidate"]["tree"])
    allowed = EQUIPMENT_PATHS + tuple(str(path) for path in inputs.get("extra_equipment_paths", []))
    if not HEX40.match(sha) or not HEX40.match(tree):
        print("inputs.json does not bind the candidate's sha and tree; nothing is run.", file=sys.stderr)
        return 2
    repo = pathlib.Path(args.equipment_repo).resolve()

    def git(*arguments, cwd=repo) -> str:
        return subprocess.run(["git", *arguments], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()

    head = git("rev-parse", "HEAD")
    refusals = [f"HEAD {head} is not this run's commit"] if os.environ.get("GITHUB_SHA", head) != head else []
    parents = git("rev-list", "--parents", "-n", "1", "HEAD").split()[1:]
    if parents != [sha]:
        refusals.append(f"the equipment commit's parents are {parents}, not exactly the candidate {sha}")
    else:
        if git("rev-parse", f"{sha}^{{tree}}") != tree:
            refusals.append(f"the candidate's tree is not {tree}")
        for line in git("diff", "--name-status", "--no-renames", sha, "HEAD").splitlines():
            status, path = line.split("\t", 1)
            if status != "A" or not path.startswith(allowed):
                refusals.append(f"the equipment commit does more than add equipment: {status} {path}")
    if refusals:
        print("\n".join(refusals), file=sys.stderr)
        return 2
    candidate = pathlib.Path(args.candidate).resolve()
    subprocess.run(["git", "clone", "--quiet", "--no-checkout", "--no-hardlinks", "-c", "core.autocrlf=false", str(repo),
                    str(candidate)], check=True)
    git("-c", "advice.detachedHead=false", "checkout", "--quiet", "--detach", sha, cwd=candidate)
    checked = {"commit": git("rev-parse", "HEAD", cwd=candidate), "tree": git("rev-parse", "HEAD^{tree}", cwd=candidate),
               "dirty": git("status", "--porcelain", "--untracked-files=all", cwd=candidate)}
    if checked != {"commit": sha, "tree": tree, "dirty": ""}:
        print(f"the candidate checkout is not exact: {checked}", file=sys.stderr)
        return 2
    evidence = pathlib.Path(args.evidence).resolve()
    evidence.mkdir(parents=True, exist_ok=True)
    (evidence / "identity.json").write_text(json.dumps({
        "candidate": {"sha": sha, "tree": tree}, "equipment_commit": head,
        "workflow_ref": os.environ.get("GITHUB_REF", ""), "run": os.environ.get("GITHUB_RUN_ID", ""),
        "attempt": os.environ.get("GITHUB_RUN_ATTEMPT", ""), "runner": os.environ.get("RUNNER_NAME", ""),
        "image": [os.environ.get("ImageOS", ""), os.environ.get("ImageVersion", "")]}, indent=2), encoding="utf-8")
    print(f"candidate {sha} tree {tree}; equipment {head}", flush=True)
    return 0


def summarize(args) -> int:
    """One report on every outcome. Reporting never changes a result: no verdict file is itself ``unknown``, by name."""
    out = pathlib.Path(args.out)
    verdict = read_json(out / "verdict.json") or {
        "status": "unknown", "unobserved": [f"no verdict was written: the matrix step ended {args.outcome}"]}
    lines = [f"### First frame, {args.name}: {verdict['status']}", f"Matrix step: {args.outcome}." + (
        "" if args.outcome == "success" else " A step's failure or ceiling says nothing of the product; only the "
                                             "contradictions below do."),
             "Component frames only; a reader must still look at the saved images."]
    if verdict.get("held_provenance"):
        lines.append(f"Frames held, by page path: {json.dumps(verdict['held_provenance'])}")
    if verdict.get("scope"):
        lines.append(f"Scope: {verdict['scope']}.")
    for title in ("contradictions", "unobserved", "appearance", "advisories", "observations"):
        reasons = verdict.get(title, [])
        lines += [f"{title}: {len(reasons)}"] + [f"- {reason}" for reason in reasons[:60]]
        if len(reasons) > 60:
            lines.append(f"{len(reasons) - 60} more entries retained in verdict.json ({title}).")
    text = "\n".join(lines) + "\n"
    out.mkdir(parents=True, exist_ok=True)
    (out / "result-summary.md").write_text(text, encoding="utf-8")
    print(text, flush=True)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as stream:
            stream.write(text)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("plan")
    report = sub.add_parser("summarize")
    for flag in ("--out", "--name", "--outcome"):
        report.add_argument(flag, required=True)
    identity = sub.add_parser("bind")
    for flag in ("--equipment-repo", "--candidate", "--evidence"):
        identity.add_argument(flag, required=True)
    one = sub.add_parser("run-case")
    one.add_argument("--case", choices=sorted(CASES), required=True)
    one.add_argument("--choice", choices=("light", "dark", "system"), required=True)
    every = sub.add_parser("matrix")
    for each in (one, every):
        each.add_argument("--source", required=True, help="the candidate's own checkout (its ouroboros/ and web/)")
        each.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if args.command == "plan":
        plans = {name: plan_for(flag) for name, flag in (("windows", True), ("linux", False))}
        print(json.dumps({name: {"hosts": hosts, "runs_per_host": runs, "runs": len(hosts) * len(runs)}
                          for name, (hosts, runs) in plans.items()}, indent=1))
        return 0
    if args.command in ("summarize", "bind"):
        return summarize(args) if args.command == "summarize" else bind(args)
    if os.environ.get("GITHUB_ACTIONS") != "true" or os.environ.get("RUNNER_ENVIRONMENT") != "github-hosted":
        print("Refusing: this opens real windows and moves host settings, so it runs only on a GitHub-hosted runner "
              "(GITHUB_ACTIONS=true, RUNNER_ENVIRONMENT=github-hosted).", file=sys.stderr)
        return 2
    return run_case(args) if args.command == "run-case" else matrix(args)


if __name__ == "__main__":
    sys.exit(main())

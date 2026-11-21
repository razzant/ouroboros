"""Observe the real pywebview 5.4 launcher from *inside* its own process on a disposable Mac.

The driver passes one private socketpair endpoint to the two source-launcher entry points.
The observer uses the test account's existing capture permissions and sends input only to its own window.
Only the fixture's launcher starts the observer. Its children use their normal entry points.
"""
from __future__ import annotations

import base64
import json
import os
import re
import secrets
import shutil
import socket
import sys
import threading
import time
from multiprocessing.connection import Connection
from pathlib import Path

LIMIT = 64 * 1024
HERE = Path(__file__).resolve().parent
OWN_FILES = ("native_window.py", "window_renewal.py", "arm_source.py", "host_processes.py", "journey_model.py", "typing_evidence.py")
CAPTURE_BRIDGE = HERE.parents[1] / "skills/unix_computer_use/macos_capture.py"
WINDOW_ID = ("shell", "pid", "window", "number")
TYPING_WORD = "Window"
KEY_CODES = {"\r": 36}  # kVK_Return: WebKit names a synthesized Return "Enter" by its key code
OSLOG_RESULTS = ("unavailable", "no_matching_entry", "entry_limit", "matched")
_WEBKIT_INFO = re.compile(rb"\d{4}-\d\d-\d\d[T ]\d\d:\d\d:\d\d[.,]\d{3}Z? \[INFO\] launcher\.webkit: WebKit ")
_REPLACEMENT = re.compile(rb"window_replacement_requested requested_at=(\d+\.\d{3})")
_LOOKUP = re.compile(rb"process_log_lookup scope=current_process since=\d+\.\d{3} until=(\d+\.\d{3}) result=(\w+)")


def launcher_diagnostics(arm_root: Path, began: float, ended: float) -> dict:
    """Read only this synthetic arm's launcher.log. Pair the request and lookup by their shared timestamp."""
    def result(outcome, requested=False):
        return {"window_replacement_requested": requested, "process_log_lookup": outcome}

    limit = 2 * 1024 * 1024  # launcher's rotating-file bound; no fallback to other logs
    try:
        path = arm_root.resolve() / "data/logs/launcher.log"
        if path.resolve() != path or not path.is_file():
            return result("launcher_log_unavailable")
        with path.open("rb") as stream:
            data = stream.read(limit + 1)
    except (OSError, RuntimeError):
        return result("launcher_log_unavailable")
    if len(data) > limit:
        return result("launcher_log_limit")
    requests, lookups = [], {}
    for line in data.splitlines():
        prefix = _WEBKIT_INFO.match(line)
        if not prefix:
            continue
        message = line[prefix.end():]
        if (match := _REPLACEMENT.fullmatch(message)) and round(began, 3) <= float(match[1]) <= round(ended, 3):
            requests.append(match[1])
        elif match := _LOOKUP.fullmatch(message):
            lookups.setdefault(match[1], []).append(match[2].decode("ascii"))
    if len(requests) != 1:
        return result("ambiguous" if requests else "request_unobserved", bool(requests))
    outcomes = lookups.get(requests[0], [])
    return result(outcomes[0] if len(outcomes) == 1 and outcomes[0] in OSLOG_RESULTS else "lookup_unobserved", True)


def same_window(left, right):
    return bool(left and right and left["number"] > 0 and all(left[k] == right[k] for k in WINDOW_ID))


def same_view(left, right):
    return same_window(left, right) and left["installed"] and right["installed"] and left["view"] == right["view"]


def visible_key_view(target, state):
    return same_view(target, state) and state["visible"] and state["key"] and state["on_screen"]


def public_reason(exc: BaseException) -> str:
    """The exception type, and its message only when the fixture or this observer raised it with its own words:
    a page, log, path, command output or third-party message stays out of the public report."""
    tb = exc.__traceback__
    while tb is not None and tb.tb_next is not None:
        tb = tb.tb_next
    own = tb is not None and Path(tb.tb_frame.f_code.co_filename).name in OWN_FILES
    if own and isinstance(exc, (RuntimeError, TimeoutError, ValueError)):
        return f"{type(exc).__name__}: {str(exc)[:160]}"
    return type(exc).__name__


class Channel:
    """One request per action. A late answer belongs to its own request, never to the next one."""

    def __init__(self, sock):
        self.conn = Connection(sock.detach())
        self.sequence = 0

    def call(self, op, **args):
        self.sequence += 1
        payload = json.dumps([self.sequence, op, args]).encode()
        if len(payload) > LIMIT:
            raise ValueError("oversized observation request")
        self.conn.send_bytes(payload)
        deadline = time.monotonic() + 35
        while self.conn.poll(max(0, deadline - time.monotonic())):
            seq, ok, value = json.loads(self.conn.recv_bytes(LIMIT))
            if seq != self.sequence:
                continue
            if not ok:
                raise RuntimeError(f"{op}: {value}")
            return value
        raise TimeoutError(f"no answer for {op}; outcome unknown")


# Everything the copied observer reads while its module body runs, not only what its calls later import.
OBSERVER_RESOURCES = ("native_window.py", "composer_clearance.py", "bubble_probe.js")


def stage_observer(root):
    """Build the arm's own observer directory: the launcher's exec loads this copy, so a resource left in
    the source tree would break ``from native_window import attach`` and leave the arm with no observer."""
    hooks = root / "observer"
    hooks.mkdir(exist_ok=True)
    for name in OBSERVER_RESOURCES:
        shutil.copyfile(HERE / name, hooks / name)
    shutil.copyfile(CAPTURE_BRIDGE, hooks / "macos_capture.py")
    return hooks


def launcher_channel(root, entry, successor, out, tag):
    """Carry the private endpoint across the launcher's own POSIX exec, never changing either tree."""
    hooks = stage_observer(root)
    (hooks / "sitecustomize.py").write_text(
        "import os, sys\nfrom pathlib import Path\n"
        f"if str(Path(sys.argv[0]).absolute()) in ({str(entry)!r}, {str(successor)!r}):\n"
        "    from native_window import attach\n    attach()\n", encoding="utf-8")
    parent, child = socket.socketpair()
    env = {"PYTHONPATH": str(hooks), "RENEWAL_OBSERVER_FD": str(child.fileno()),
           "RENEWAL_PICTURES": str(out), "RENEWAL_TAG": tag}
    return Channel(parent), child, env


def attach():
    fd = int(os.environ["RENEWAL_OBSERVER_FD"])
    conn = Connection(fd)
    os.set_inheritable(fd, True)  # same launcher, new program image; Popen children close other descriptors
    threading.Thread(target=serve, args=(conn,), name="window-observation", daemon=True).start()
    # pywebview 5.4 calls before_show synchronously, before loading the first URL.
    # A fresh source process installs the same witness. In-process renewal shares this configuration.
    import webview

    create = webview.create_window
    def watched_window(*args, **kwargs):
        window = create(*args, **kwargs)
        if type(kwargs.get("js_api")).__name__ == "MainApi":
            webview.create_window = create
            window.events.before_show += install_send_counter
        return window
    webview.create_window = watched_window


def install_send_counter(window):
    import WebKit
    from webview.platforms.cocoa import BrowserView

    script = """
        if (location.hostname === '127.0.0.1') {
            const tag = TAG, key = 'renewal-wire-' + tag;
            const labels = ['Hello', 'Held', 'Project-seed', 'Project-update', 'Project-handoff'].map(s => s + '-' + tag);
            const read = () => JSON.parse(localStorage.getItem(key) || 'null') ?? labels.map(() => 0);
            const send = WebSocket.prototype.send;
            WebSocket.prototype.send = function(raw) {
                try {
                    const f = JSON.parse(raw);
                    const i = labels.indexOf(f.content);
                    if (f.type === 'chat' && i >= 0) {
                        const counts = read();
                        counts[i]++;
                        localStorage.setItem(key, JSON.stringify(counts));
                    }
                } catch (_) {}
                return send.call(this, raw);
            };
            window.__renewalWireCounts = text => text ? read()[labels.indexOf(text)] : read().slice(0, 2);
        }
    """.replace("TAG", json.dumps(os.environ["RENEWAL_TAG"]))
    controller = BrowserView.instances[window.uid].webview.configuration().userContentController()
    controller.addUserScript_(WebKit.WKUserScript.alloc().initWithSource_injectionTime_forMainFrameOnly_(
        script, WebKit.WKUserScriptInjectionTimeAtDocumentStart, True))


def serve(conn):
    host = None
    session = secrets.token_hex(6)
    while True:
        try:
            sequence, op, args = json.loads(conn.recv_bytes(LIMIT))
        except (EOFError, OSError):
            return
        try:
            cocoa = sys.modules.get("webview.platforms.cocoa")
            if host is None and cocoa and hasattr(cocoa, "BrowserView"):
                host = AttachedWindow(cocoa.BrowserView, session)
            if host is None:
                if op != "state":
                    raise RuntimeError("the launcher has not opened its window")
                result = None
            else:
                result = host.observe(op, args)
            answer = json.dumps([sequence, True, result]).encode()
            if len(answer) > LIMIT:
                raise ValueError("oversized observation result")
        except Exception as exc:
            answer = json.dumps([sequence, False, public_reason(exc)]).encode()
        try:
            conn.send_bytes(answer)
        except (OSError, EOFError):
            return


class AttachedWindow:
    def __init__(self, browser_class, session):
        import AppKit
        import Foundation
        import WebKit
        import objc
        from PyObjCTools import AppHelper

        self.AppKit, self.Foundation, self.WebKit = AppKit, Foundation, WebKit
        self.objc, self.AppHelper, self.browsers = objc, AppHelper, browser_class
        self.session, self.identities = session, {}
        self.out = Path(os.environ["RENEWAL_PICTURES"])
        self.compositor = None

    # Reuse the maintained AppKit rendezvous and WK snapshot callbacks. Do not call NativeHost's
    # constructor: the launcher already created the window, configured the renderer and owns its delegates.
    def on_main(self, call):
        from composer_clearance import NativeHost

        return NativeHost.on_main(self, call)

    @property
    def browser(self):
        wrapper = getattr(sys.modules["__main__"], "_webview_window", None)
        return self.browsers.instances.get(wrapper.uid) if wrapper else None

    @property
    def webview(self):
        return self.browser.webview

    def identity(self, name, obj):
        # Keep labels for addresses, without retaining objects. Returning to the same object returns its old label.
        pointer = self.objc.pyobjc_id(obj)
        labels = self.identities.setdefault(name, {})
        return f"{self.session}/{name}/{labels.setdefault(pointer, len(labels) + 1)}"

    def state(self):
        from ouroboros import launcher_background

        browser = self.browser
        if browser is None or not browser.webview:
            return None
        window, view = browser.window, browser.webview
        rect = window.frame()
        client = view.convertRect_toView_(view.bounds(), None)  # NSWindow base coords have their origin at bottom left
        url = view.URL()
        background = getattr(launcher_background._active, "background", None)
        chrome = background.chrome if background and getattr(background.window, "native", None) == window else None
        return {"shell": self.session, "window": self.identity("window", window), "view": self.identity("view", view),
                "installed": bool(window.isKindOfClass_(self.AppKit.NSWindow) and view.isKindOfClass_(self.WebKit.WKWebView)
                                  and window.contentView() == view and view.window() == window),
                "number": int(window.windowNumber()), "visible": bool(window.isVisible()), "key": bool(window.isKeyWindow()),
                "on_screen": bool(window.screen() is not None and window.occlusionState() & self.AppKit.NSWindowOcclusionStateVisible),
                "fullscreen": bool(window.styleMask() & self.AppKit.NSWindowStyleMaskFullScreen),
                "fullscreen_phase": getattr(chrome, "_phase", None),
                "frame": [rect.origin.x, rect.origin.y, rect.size.width, rect.size.height],
                "viewport": [client.origin.x, rect.size.height - client.origin.y - client.size.height,
                             client.size.width, client.size.height],
                "zoom": float(view.pageZoom()), "port": int(url.port() or 0) if url else 0,
                "loopback": bool(url and url.host() == "127.0.0.1" and url.scheme() == "http"),
                "home": bool(url and (url.path() or "/") == "/"),
                "builtin_setup": bool(url and url.path() == "/onboarding" and url.query() == "builtin=1"), "pid": os.getpid()}

    def require_room(self, target):
        state = self.state()
        if not (visible_key_view(target, state) and state["loopback"] and state["home"]
                and all(state[k] == target[k] for k in ("port", "frame", "viewport", "zoom"))):
            raise RuntimeError("the installed view, room, key window or capture geometry changed")

    def page(self, script, data):
        done, box = threading.Event(), {}
        def finished(result, error):
            box.update(result=result, error=error is not None)
            done.set()
        def start():
            self.webview.callAsyncJavaScript_arguments_inFrame_inContentWorld_completionHandler_(
                "const a = JSON.parse(payload);return JSON.stringify(await (async () => {" + script + "})() ?? null)",
                {"payload": json.dumps(data)}, None, self.WebKit.WKContentWorld.pageWorld(), finished)
        self.on_main(start)
        if not done.wait(14):
            raise TimeoutError("WebKit did not answer the document operation")
        if box["error"]:
            raise RuntimeError("WebKit rejected the document operation")
        return json.loads(str(box["result"]))

    def key(self, text, target):
        if target is not None:
            self.require_room(target)
        window = self.browser.window
        view = self.webview
        if not window.isVisible() or not window.isKeyWindow() or view.window() != window:
            raise RuntimeError("the target is not the visible key window")
        responder = window.firstResponder()
        if responder != view and not (hasattr(responder, "isDescendantOf_") and responder.isDescendantOf_(view)):
            if not window.makeFirstResponder_(view):
                raise RuntimeError("WebKit did not take keyboard focus")
        for char in text:
            for kind in (self.AppKit.NSEventTypeKeyDown, self.AppKit.NSEventTypeKeyUp):
                event = self.AppKit.NSEvent.keyEventWithType_location_modifierFlags_timestamp_windowNumber_context_characters_charactersIgnoringModifiers_isARepeat_keyCode_(
                    kind, (0, 0), 0, time.monotonic(), window.windowNumber(), None, char, char, False, KEY_CODES.get(char, 0))
                window.sendEvent_(event)
        return True

    @staticmethod
    def choose(menu, title):
        index = menu.indexOfItemWithTitle_(title) if menu is not None else -1
        if index < 0 or not menu.itemAtIndex_(index).isEnabled():
            raise RuntimeError(f"no enabled {title} menu item")
        menu.performActionForItemAtIndex_(index)
        return True

    def action(self, kind, args):
        window = self.browser.window
        app = self.AppKit.NSApplication.sharedApplication()
        if kind in ("hide", "reopen"):
            from ouroboros import launcher_background as background

            indicator = background._active
            item = getattr(indicator, "_item", None)
            if item is None or not indicator.ready.is_set() or background.keep_running_choice() != "true":
                raise RuntimeError("Keep running or the launcher's menu-bar item is unavailable")
            if kind == "reopen":
                return self.choose(item.menu(), "Open Ouroboros")
            window.performClose_(None)
        elif kind == "resize":
            window.setContentSize_((args["width"], args["height"]))
        elif kind == "fullscreen":
            window.toggleFullScreen_(None)
        elif kind == "zoom":
            from ouroboros.launcher_page_zoom import view_menu

            return self.choose(view_menu(app.mainMenu()), args["title"])
        elif kind == "help":
            title = "Open built-in setup"
            for item in app.mainMenu().itemArray():
                menu = item.submenu()
                if menu and menu.itemWithTitle_(title):
                    return self.choose(menu, title)
            raise RuntimeError("the Help menu action is unavailable")
        elif kind == "quit":
            # AppKit delivers the ordinary application Quit to this process's installed delegate.
            self.AppHelper.callAfter(app.terminate_, None)
        else:
            raise ValueError("unknown window action")
        return True

    def arm_cancel(self, prompt):
        """Select only Cancel on this test's own JS confirmation during AppKit's nested modal loop."""
        ns = self.Foundation
        app = self.AppKit.NSApplication.sharedApplication()
        deadline = time.monotonic() + 12

        def tick(timer):
            modal = app.modalWindow()
            if modal is not None:
                stack = [modal.contentView()]
                words, cancel = [], None
                while stack:
                    view = stack.pop()
                    stack.extend(view.subviews())
                    if hasattr(view, "stringValue"):
                        words.append(str(view.stringValue()))
                    if isinstance(view, self.AppKit.NSButton) and view.title() == "Cancel":
                        cancel = view
                if prompt in words and cancel is not None:
                    timer.invalidate()
                    self.cancel_clicked = True
                    cancel.performClick_(None)
                    return
                if prompt in words and time.monotonic() >= deadline:
                    app.abortModal()  # only this synthetic alert; absence of the Cancel action remains a gap
            if time.monotonic() >= deadline:
                timer.invalidate()

        self.cancel_clicked = False
        timer = ns.NSTimer.timerWithTimeInterval_repeats_block_(0.1, True, tick)
        for mode in (ns.NSRunLoopCommonModes, self.AppKit.NSModalPanelRunLoopMode):
            ns.NSRunLoop.mainRunLoop().addTimer_forMode_(timer, mode)
        return True

    def window_image(self, target):
        import Quartz
        from macos_capture import Bridge, await_handler, pixel_size

        number = target["number"]
        quartz_capture = getattr(Quartz, "CGWindowListCreateImage", None)
        if self.compositor != "ScreenCaptureKit" and quartz_capture is not None:
            try:
                image = quartz_capture(Quartz.CGRectNull, Quartz.kCGWindowListOptionIncludingWindow, number,
                                       Quartz.kCGWindowImageBoundsIgnoreFraming)
            except (AttributeError, NotImplementedError):
                image = None
            if image is not None:
                self.compositor = "CoreGraphics"
                return image
            if self.compositor == "CoreGraphics":
                raise RuntimeError("WindowServer stopped returning the selected window")

        # Modern macOS can withhold the legacy API. This path also filters to one window, with an existing grant.
        if self.compositor is None:
            bridge = Bridge()
            if bridge.screen_recording() != "granted":
                raise RuntimeError("window capture unavailable without an existing Screen Recording grant")
            if bridge.load():
                raise RuntimeError("ScreenCaptureKit window capture is unavailable")
            self.capture_bridge, self.compositor = bridge, "ScreenCaptureKit"
        bridge = self.capture_bridge
        deadline = time.monotonic() + 10
        def wait(start):
            arrived, result, error = await_handler(start, deadline, pump=time.sleep)  # AppKit has its own main loop
            if not arrived or result is None or error is not None:
                raise RuntimeError("ScreenCaptureKit did not return the requested window observation")
            return result
        content = wait(bridge.shareable_content)
        selected = next((w for w in content.windows() if w.windowID() == number and w.isOnScreen()
                         and w.owningApplication().processID() == target["pid"]), None)
        if selected is None:
            raise RuntimeError("the installed window was not available for composition")
        single = bridge.content_filter(selected)
        size = pixel_size(*bridge.content_size(single))
        config, applied = bridge.configuration(*size)
        if hasattr(config, "setIncludeChildWindows_"):
            config.setIncludeChildWindows_(False)
            if config.includeChildWindows():
                raise RuntimeError("could not exclude child windows")
        if applied["cursor_included"] or applied["shadow_included"] or min(size) < 1:
            raise RuntimeError("single-window capture geometry unavailable")
        return wait(lambda done: bridge.screenshot(single, config, done))

    def capture(self, kind, name, target=None):
        if not re.fullmatch(r"[a-z0-9-]{1,48}", name):
            raise ValueError("invalid capture name")
        self.on_main(lambda: self.require_room(target))
        path = self.out / f".{name}.pending"
        if kind == "renderer":
            from composer_clearance import NativeHost

            NativeHost.snapshot(self, path, timeout=12)
        elif kind == "window":
            image = self.window_image(target)
            rep = self.AppKit.NSBitmapImageRep.alloc().initWithCGImage_(image)
            data = rep.representationUsingType_properties_(self.AppKit.NSBitmapImageFileTypePNG, {})
            if not data or data.length() > 16 << 20:
                raise RuntimeError("no bounded window image")
            path.write_bytes(bytes(data))
        else:
            raise ValueError("unknown capture kind")
        self.on_main(lambda: self.require_room(target))
        if not 0 < path.stat().st_size <= 16 << 20:
            raise RuntimeError("empty or oversized window observation")
        photo = path.replace(self.out / f"{name}.png")
        return {"file": photo.name, "source": self.compositor if kind == "window" else "WKWebView"}

    def observe(self, op, args):
        if op == "page":
            return self.page(args["script"], args["data"])
        if op == "capture":
            return self.capture(args["kind"], args["name"], args["target"])
        if op == "state":
            return self.on_main(self.state)
        if op == "key":
            return self.on_main(lambda: self.key(args["text"], args.get("target")))
        if op == "action":
            return self.on_main(lambda: self.action(args["kind"], args))
        if op == "arm_cancel":
            return self.on_main(lambda: self.arm_cancel(args["prompt"]))
        if op == "cancel_clicked":
            return getattr(self, "cancel_clicked", False)
        if op == "slots":
            def read():
                owner = getattr(sys.modules["__main__"], "_main_renderer", None)
                if owner is None:
                    return None
                with owner.session.lock:
                    slots = owner.session.items
                    marker = args["marker"]
                    encoded = base64.b64encode(marker.encode()).decode().rstrip("=")
                    return {"files": slots.get("ouro_chat_files"), "text": slots.get("ouro_chat_draft"),
                            "project": slots.get("ouro_chat_draft:" + str(args.get("room", ""))),
                            "project_files": slots.get("ouro_chat_files:" + str(args.get("room", ""))),
                            "reply": json.loads(slots.get("ouro_chat_reply", "null")),
                            "project_reply": json.loads(slots.get("ouro_chat_reply:" + str(args.get("room", "")), "null")),
                            "frame": json.loads(slots.get("ouro_chat_unconfirmed", "[]")),
                            "contains_fixture_bytes": any(marker in v or encoded in v for v in slots.values())}
            return self.on_main(read)
        raise ValueError("unknown observer operation")


# The page-side probe of one user bubble (bubble_probe.js, the body of an async function over ``a``): effective
# display, visibility and opacity, the feed's clip, the viewport and an exposed hit point. ``visibility`` is a fixed word.
BUBBLE_PROBE = (Path(__file__).with_name("bubble_probe.js")).read_text(encoding="utf-8")


class RoomWindow:
    """Actions on the shipped document in the launcher's currently installed view."""

    def __init__(self, channel, wait, out, observe):
        self.channel, self.wait, self.out, self.observe = channel, wait, out, observe

    def call(self, op, **args):
        return self.channel.call(op, **args)

    def page(self, script, **data):
        return self.call("page", script=script, data=data)

    def state(self):
        return self.call("state")

    def action(self, kind, **args):
        return self.call("action", kind=kind, **args)

    def until(self, read, seconds=20):
        return self.wait(read, seconds, 0.2)

    def key_window(self, target):
        return target and self.until(lambda: (s := self.state()) and visible_key_view(target, s) and s)

    def visible(self, selector, *, editable=False):
        return self.page("""
            const e = document.querySelector(a.sel);
            if (!e?.isConnected) return a.editable ? null : false;
            if (!e.getClientRects().length) return false;
            if (!a.editable) return true;
            const r = e.getBoundingClientRect();
            return !e.disabled && !e.readOnly && e.dataset.desktopRoomOpening !== '1' && !e.closest('[inert]') &&
                document.elementFromPoint(r.x + r.width/2, r.y + r.height/2) === e;
        """, sel=selector, editable=editable)

    def click(self, selector):
        return self.page("const e=document.querySelector(a.sel); if (!e || e.disabled) return false;"
                         "e.click(); return true", sel=selector)

    def main(self):
        if self.page("return document.querySelector('#project-panel')?.classList.contains('open')"):
            self.click("#project-panel-close")
            if not self.until(lambda: self.page("return !document.querySelector('#project-panel.open')")):
                return False
        if not self.visible("#chat-input"):
            self.click('[data-nav-page="chat"]')
        return self.until(lambda: self.visible("#chat-input"))

    def open_project(self, project):
        field = f'#project-panel.open #pchat-{project["id"]}-input'
        if self.visible(field):
            return True  # clicking the selected sidebar row would close this Project
        row = f'.nav-project-row[data-project-id="{project["id"]}"]'
        if not self.until(lambda: self.page("return !!document.querySelector(a.sel)", sel=row)) or not self.click(row):
            raise RuntimeError("project navigation row unavailable")
        if not self.until(lambda: self.visible(field)):
            raise RuntimeError("project composer unavailable")
        return True

    def prepare_draft(self, staged, marker, project=None):
        if not (self.open_project(project) if project is not None else self.main()):
            raise RuntimeError("the room was unavailable while preparing its draft")
        prefix, text, filename = staged["prefix"], staged["text"], staged["filename"]
        field = prefix + "-input"
        if filename:
            self.page("for (const b of document.querySelectorAll(a.remove)) b.click()",
                      remove=prefix + "-attachment-preview .attach-remove")
        typed = self.type(field, text)
        if filename:
            reply = staged["reply"]
            staged["quoted"] = bool(reply and self.quote(reply, prefix) and self.follows_quote(reply, prefix))
            staged["room_id"] = project["chat_id"] if project is not None else None
            staged["saved_quote"] = self.stored_quote(marker, staged["room_id"]) if staged["quoted"] else None
            staged["shown_quote"] = self.composer_quote(prefix) if staged["quoted"] else None
            self.attach(filename, marker, prefix)
        doc = self.draft(field)
        if doc:
            staged["typed"] = typed and doc["text"] == text
            staged["file"] = doc["names"] == ([filename] if filename else []) and not doc["notice"]
            if not filename:
                staged["file"] = staged["file"] and doc["input_files"] == 0
                staged["quoted"] = not doc["reply"]

    def inspect_draft(self, staged, checks, marker):
        """Inspect the room already on screen, before its typing or Send changes the draft."""
        prefix, reply = staged["prefix"], staged["reply"]
        field, project = prefix + "-input", prefix != "#chat"
        def restored():
            doc = self.draft(field)
            return doc if doc and doc["text"] == staged["text"] and (not reply or doc["reply"]) else None
        doc = self.until(restored, 8) or self.draft(field)
        if doc is None:
            return
        if staged.get("typed"):
            if project and not staged["filename"]:
                checks["project_draft"] = doc["text"] == staged["text"] and not any(
                    doc[k] for k in ("reply", "names", "notice", "input_files"))
            else:
                checks["project_draft" if project else "draft_restored"] = doc["text"] == staged["text"]
        key = "project_" if project else ""
        if reply and staged.get("quoted"):
            checks[key + "reply_restored"] = self.observe("restored quote in " + prefix, lambda:
                self.restored_quote(staged, marker, reply, prefix))
        if staged["filename"] and staged.get("file"):
            notice = not doc["names"] and doc["input_files"] == 0 and "Attach your file again" in doc["notice"]
            checks[key + "reattach_notice"] = notice
            if notice and not project:
                self.attach(staged["filename"], marker, prefix)
                after = self.draft(field)
                checks["attachment_restaged"] = after["names"] == [staged["filename"]] and not after["notice"]

    def stored_quote(self, marker, room=None):
        """The room's complete staged quote exactly as the product stores it, compared whole in
        memory: a restored quote that swapped its reference, author, time, snippet, unresolved
        state or attachment-view identity (the media view is the product's own safe upload
        reference, never bytes) is a demonstrated difference. Nothing here reaches the report —
        the recorded checks carry booleans only, never a quote field or byte."""
        slots = self.custody(marker, room)
        quote = (slots or {}).get("project_reply" if room is not None else "reply")
        return quote if isinstance(quote, dict) else None

    def composer_quote(self, prefix):
        """Read the mounted strip's visible author, time and text, independently of native storage."""
        return self.page("""
            const field = document.querySelector(a.prefix + '-input');
            if (!field?.getClientRects().length || field.dataset.desktopRoomOpening === '1') return null;
            const strip = document.querySelector(a.prefix + '-input-reply');
            const visible = e => e?.getClientRects().length && getComputedStyle(e).visibility === 'visible';
            const quote = strip?.closest('.chat-quote--composer');
            if (!visible(strip) || !quote) return false;
            const part = name => {
                const e = quote.querySelector('.chat-quote-' + name);
                return visible(e) ? e.innerText : null;
            };
            return {author:part('author'), when:part('when'), text:part('text'),
                label:strip.getAttribute('aria-label'), thumb:!!visible(quote.querySelector('.chat-quote-thumb'))};
        """, prefix=prefix)

    @staticmethod
    def quote_display(shown, *, expect_time):
        """Compare the visible local clock, not the day wording, in both strip and label."""
        if not isinstance(shown, dict) or not all(shown.get(k) for k in ("author", "text", "label")):
            return None
        when = shown.get("when")
        if when is None and not expect_time:
            return shown
        # formatMsgTime always ends its short form in HH:MM. Leave the localized prefix
        # uninterpreted; the persisted timestamp is compared exactly before this check.
        clock = re.search(r"(?<![0-9])(?:[01][0-9]|2[0-3]):[0-5][0-9]$", when or "")
        head, sep, snippet = shown["label"].partition(": ")
        suffix = ", " + when.removeprefix("· ") if clock else ""
        if not clock or not sep or not head.endswith(suffix):
            return None
        return dict(shown, when=clock[0], label=head.removesuffix(suffix) + ", " + clock[0] + sep + snippet)

    def restored_quote(self, staged, marker, address, prefix):
        """Require the complete stored shape, the on-screen strip and a jump to the original row."""
        expected, current = staged.get("saved_quote"), self.stored_quote(marker, staged.get("room_id"))
        if expected is None or current is None:
            return None
        if current != expected or expected.get("address") != address:
            return False
        shown = self.composer_quote(prefix)
        expect_time = bool(expected.get("ts"))
        before = RoomWindow.quote_display(staged.get("shown_quote"), expect_time=expect_time)
        if before is None or shown is None:
            return None
        if RoomWindow.quote_display(shown, expect_time=expect_time) != before:
            return False
        return self.follows_quote(address, prefix, staged.get("quote_text"))

    def shows_room(self, project, target, *, in_list=False, origin=None):
        """Read Main, a Project, or its sidebar row in place, without selecting it."""
        if not self.key_window(target):
            raise RuntimeError("the room window was not visible and key")
        def read(final=False):
            return self.page("""
            if (a.origin !== null && performance.timeOrigin !== a.origin) return null;
            const id = CSS.escape(a.id);
            const room = document.querySelector(a.list ? '.nav-project-row[data-project-id=' + id + ']' : '#project-panel.open');
            const title = room?.querySelector(a.list ? '.nav-row-label' : '#project-panel-title');
            const visible = (element, owner = element) => {
                const r = element?.getBoundingClientRect();
                return r?.width > 0 && r.height > 0 && owner.contains(document.elementFromPoint(r.x+r.width/2, r.y+r.height/2));
            };
            if (a.list) return title?.textContent === a.name && visible(title, room);
            const mounted = e => visible(e) && e.dataset.desktopRoomOpening !== '1' && !e.closest('[inert]');
            if (room) {
                if (!title?.textContent || !visible(title, title.parentElement) ||
                    !mounted(room.querySelector('textarea[id^="pchat-"][id$="-input"]'))) return null;
                if (a.name && title.textContent === a.name && room.querySelector('#pchat-' + id + '-input')) return true;
                return a.final ? false : null;
            }
            const main = document.querySelector('#chat-input');
            if (mounted(main)) return !a.name ? true : (a.final ? false : null);
            return null;
        """, id=project["id"] if project else "", name=project["name"] if project else None,
                 list=in_list, final=final, origin=origin)
        shown = self.until(read, 35 if in_list else 20)
        if not shown and not in_list:
            shown = read(final=True)  # a visible different room is a finding; an unfinished mount stays unknown
        if not visible_key_view(target, self.state()):
            raise RuntimeError("the active window changed while observing the room")
        return shown

    def watch_project(self, name):
        """ws.js dispatches this event from socket.onmessage. The sidebar poll only refreshes the list."""
        return self.until(lambda: self.page("""
            const ws = window.__ouroWs;
            if (!ws) return null;
            if (!ws.isConnected()) return false;
            const watch = window.__renewalPeer = {frame:null};
            watch.stop = ws.on('projects_changed', ({type, project_id, chat_id}) => {
                if (project_id === a.id) watch.frame = {type, project_id, chat_id};
            });
            return true;
        """, id=name))

    def value(self, field):
        return self.page("return document.querySelector(a.sel)?.value ?? null", sel=field)

    def type(self, field, value, *, replace=True, target=None):
        ready = self.until(lambda: self.visible(field, editable=True)) and self.page("""
            const e = document.querySelector(a.sel);
            if (!e || !e.getClientRects().length || e.disabled || e.readOnly) return false;
            e.focus();
            if (a.replace) e.select(); else e.setSelectionRange(e.value.length, e.value.length);
            return document.activeElement === e;
        """, sel=field, replace=replace)
        if not ready:
            raise RuntimeError("the composer could not take focus")
        expected = value if replace else self.value(field) + value
        self.call("key", text=value, target=target)
        return bool(self.until(lambda: self.value(field) == expected))

    def quote(self, address, prefix="#chat"):
        return self.until(lambda: self.page("""
            const bubble = [...document.querySelectorAll(a.prefix + '-messages [data-row-address]')]
                .find(row => row.dataset.rowAddress === a.address);
            const button = bubble?.querySelector('[data-row-reply]');
            if (!button) return false;
            button.click();
            return true;
        """, address=address, prefix=prefix))

    def follows_quote(self, address, prefix="#chat", quote_text=None):
        # A highlight from the preceding jump lasts briefly. Only a fresh one witnesses this button's destination.
        highlighted = prefix + "-messages .is-quote-target"
        if not self.until(lambda: not self.page("return !!document.querySelector(a.sel)", sel=highlighted), 4):
            raise RuntimeError("the previous quote highlight did not clear")
        button = prefix + "-input-reply"
        if not self.visible(button):
            return False
        # A Project restoring Main's quote must not take the fixture to Main when the strip is pressed.
        if quote_text and not self.page("return document.querySelector(a.sel)?.getAttribute('aria-label')?.endsWith(a.text)",
                                        sel=button, text=quote_text):
            return False
        if not self.click(button):
            return False
        return bool(self.until(lambda: self.page("""
            return [...document.querySelectorAll(a.sel)]
                .some(row => row.dataset.rowAddress === a.address)
        """, address=address, sel=highlighted), 3))

    def draft(self, field="#chat-input"):
        return self.page("""
            const field = document.querySelector(a.field), el = suffix => document.querySelector(a.prefix + suffix);
            const preview = el('-attachment-preview'), picker = el('-file-input');
            if (!field?.getClientRects().length || !preview || !picker) return null;
            const notice = preview.querySelector('[role=status]');
            return {text:field.value, reply:!!el('-input-reply')?.getClientRects().length, input_files:picker.files.length,
                names:[...preview.querySelectorAll('.attach-name')].map(n=>n.textContent),
                notice:notice?.getClientRects().length ? notice.textContent : ''};
        """, field=field, prefix=field.removesuffix("-input"))

    def attach(self, name, marker, prefix="#chat"):
        return self.page("""
            const input = document.querySelector(a.picker);
            if (!input || input.disabled) return false;
            const dt = new DataTransfer();
            dt.items.add(new File([a.marker], a.name, {type:'text/plain'}));
            input.files = dt.files;
            input.dispatchEvent(new Event('change', {bubbles:true}));
            return true;
        """, name=name, marker=marker, picker=prefix + "-file-input")

    def send_once(self, text, hold, prefix="#chat", target=None, *, from_draft=False):
        # Let the product create its own id, local echo, send outcome and retained frame. The
        # held frame stops at exactly one transport write. All other frames go to the real socket.
        field = prefix + "-input"
        if from_draft:
            if not self.visible(field, editable=True):
                return None
        elif not self.type(field, text, target=target):
            raise RuntimeError("the original message was not typed")
        if not self.until(lambda: self.page("return window.__ouroWs?.isConnected() === true")):
            return None
        return self.page("""
            const proto = WebSocket.prototype, original = proto.send;
            let seen = null;
            proto.send = function(raw) {
                let msg;
                try { msg = JSON.parse(raw); } catch (_) {}
                if (msg?.type === 'chat' && msg.content === a.text) {
                    seen = Object.fromEntries(['type','content','client_message_id','sender_session_id',
                        'chat_id','project_id','reply_to'].filter(k => k in msg).map(k => [k,msg[k]]));
                    proto.send = original;
                    if (a.hold) return;
                }
                return original.call(this, raw);
            };
            try {
                const field = document.querySelector(a.field), button = document.querySelector(a.button);
                const r = button?.getBoundingClientRect();
                if (!field || field.disabled || field.readOnly || field.value !== a.text || button?.disabled ||
                    (!a.fromDraft && field !== document.activeElement) ||
                    !r?.width || !r.height || !button.contains(document.elementFromPoint(r.x+r.width/2, r.y+r.height/2))) return null;
                button.click();
                for (let i=0; i<80 && !seen; i++) await new Promise(r=>setTimeout(r,50));
                return seen;
            } finally { proto.send = original; }
        """, text=text, hold=hold, fromDraft=from_draft, field=field, button=prefix + "-send")

    def send_native(self, text, prefix, target, *, from_draft):
        """One ordinary Send driven by a single AppKit key action on the observed window, through
        the product's own Enter path. Once the field is armed there is no substitution and no
        retry: native input is certified only by the key action's typed answer beside its
        captured frame, and a failed, unknown or uncaptured action leaves the Send's proofs
        missing. A frame that arrived beside an uncertain key result is kept for the accepted
        message ledger without certifying the drive, and a write whose frame was not captured
        is never repeated. Returns {wire, drive, disclosure[, writes]}: drive is "appkit" or
        None."""
        field = prefix + "-input"
        if not from_draft and not self.type(field, text, target=target):
            raise RuntimeError("the original message was not typed")
        if not self.until(lambda: self.page("return window.__ouroWs?.isConnected() === true")):
            return {"wire": None, "drive": None, "disclosure": "socket_not_connected"}
        armed = self.page("""
            if (window.__renewalSend) return null;
            const field = document.querySelector(a.field), button = document.querySelector(a.button);
            const r = button?.getBoundingClientRect();
            if (!field || field.disabled || field.readOnly || field.value !== a.text || button?.disabled ||
                !r?.width || !r.height || !button.contains(document.elementFromPoint(r.x+r.width/2, r.y+r.height/2))) return false;
            field.focus();
            if (document.activeElement !== field) return false;
            const proto = WebSocket.prototype, original = proto.send;
            const state = window.__renewalSend = {seen: null, restore: () => { proto.send = original; }};
            proto.send = function(raw) {
                try {
                    const msg = JSON.parse(raw);
                    if (msg?.type === 'chat' && msg.content === a.text) {
                        state.seen = Object.fromEntries(['type','content','client_message_id','sender_session_id',
                            'chat_id','project_id','reply_to'].filter(k => k in msg).map(k => [k,msg[k]]));
                        proto.send = original;
                    }
                } catch (_) {}
                return original.call(this, raw);
            };
            return true;
        """, field=field, button=prefix + "-send", text=text)
        if armed is not True:
            return {"wire": None, "drive": None,
                    "disclosure": "field_ineligible" if armed is False else "capture_unavailable"}
        disclosure = None
        driven = False
        try:
            driven = self.call("key", text="\r", target=target) is True  # the typed effect, never a mere quiet return
        except Exception as exc:  # noqa: BLE001 - a lost key action's outcome is unknown; nothing is ever retried
            disclosure = public_reason(exc)
        seen = self.until(lambda: self.page("return window.__renewalSend?.seen ?? null"), 8)
        if seen is not None:
            self.disarm_send_capture()
            return {"wire": seen, "drive": "appkit" if driven else None,
                    "disclosure": None if driven else (disclosure or "key_outcome_uncertain")}
        wrote = self.until(lambda: self.page("return window.__renewalWireCounts?.(a.text) ?? null", text=text), 4)
        self.disarm_send_capture()
        if wrote is not None:  # a readable counter: 0 proves no write, a positive count a write without its frame
            return {"wire": None, "drive": "appkit" if driven else None,
                    "disclosure": "frame_not_captured" if wrote else (disclosure or "native_key_did_not_submit"),
                    "writes": wrote}
        return {"wire": None, "drive": "appkit" if driven else None,
                "disclosure": disclosure or "write_counter_unavailable"}

    def disarm_send_capture(self):
        return self.page("window.__renewalSend?.restore?.(); window.__renewalSend = null; return true")

    def custody(self, marker, room):
        slots = self.call("slots", marker=marker, room=room)
        if slots is None:  # older arm has tab storage
            slots = self.page("""
                const get = k => sessionStorage.getItem(k);
                return {text:get('ouro_chat_draft'), reply:JSON.parse(get('ouro_chat_reply') || 'null'),
                    project:get('ouro_chat_draft:'+a.room), frame:JSON.parse(get('ouro_chat_unconfirmed') || '[]'),
                    project_reply:JSON.parse(get('ouro_chat_reply:'+a.room) || 'null')};
            """, room=room)
        return slots

    def bubble(self, msg_id, prefix="#chat", *, address=None):
        """The bubbles of one client message. With ``address``, ``visible_accepted`` is True only for exactly one
        bubble bound to that row address that is actually shown, False for a demonstrated duplicate, absent row or a
        different address, and None where the row is present but its visibility is not established."""
        return self.page(BUBBLE_PROBE, id=msg_id, feed=prefix + '-messages', address=address,
                         sel=prefix + '-messages .chat-bubble.user[data-client-message-id]')

    def confirm_cancel(self, prompt):
        self.call("arm_cancel", prompt=prompt)
        accepted = self.page("return window.confirm(a.prompt)", prompt=prompt)
        if not self.call("cancel_clicked"):
            raise RuntimeError("the fixture's Cancel button could not be identified")
        return accepted is False

    def document(self, port):
        """When and how this document loaded. The product's own reload (ws.js) carries ``_ouro_refresh`` and maybe
        ``_ouro_reason`` in its navigation URL, which layers.js later strips from the address bar."""
        return self.until(lambda: self.page("""
            if (location.origin !== 'http://127.0.0.1:' + a.port || location.pathname !== '/' ||
                document.readyState !== 'complete' || !window.__ouroWs) return null;
            const n = performance.getEntriesByType('navigation')[0], q = new URL(n?.name || location.href).searchParams;
            return {origin:performance.timeOrigin, type:n?.type ?? null, self_refresh:q.has('_ouro_refresh')};
        """, port=port))

    def box(self, field="#chat-input"):
        return self.page("""
            const field = document.querySelector(a.field);
            if (!field || document.activeElement !== field) return null;
            const r = field.getBoundingClientRect(), s = getComputedStyle(field);
            const ctx = document.createElement('canvas').getContext('2d');
            ctx.font = [s.fontStyle, s.fontWeight, s.fontSize, s.fontFamily].join(' ');
            const start = parseFloat(s.paddingLeft), width = ctx.measureText(a.word).width;
            const advance = ctx.measureText(a.word + ' ').width;
            if (s.direction !== 'ltr' || start + advance + width > field.clientWidth - parseFloat(s.paddingRight))
                return null;
            return [r.left+parseFloat(s.borderLeftWidth), r.top+parseFloat(s.borderTopWidth),
                    r.right-parseFloat(s.borderRightWidth), r.bottom-parseFloat(s.borderBottomWidth),
                    innerWidth, innerHeight, start, width, advance];
        """, field=field, word=TYPING_WORD)

    def typing(self, successor, field, label, checks, evidence):
        """Measure the actual focused room field in both the compositor and renderer."""
        from devtools.macos_wkwebview.typing_evidence import echo_overlap, pixels_readable, typing_delta, window_colors

        target = self.key_window(successor)
        if not target or target["port"] != successor["port"]:
            raise RuntimeError("the successor was not the active target for typing")
        key = "" if field == "#chat-input" else "project_"
        if not self.type(field, TYPING_WORD, target=target):
            checks[key + "typed"] = False
            return
        box = self.box(field)
        if not box:
            raise RuntimeError("focused composer geometry unavailable")
        evidence.update(target=target, field=field, css_field=box)
        pictures = {"window": [], "renderer": []}
        def pair(sample):
            for kind, shots in pictures.items():
                shots.append(self.observe(f"{label} {kind} {sample}", lambda: self.call(
                    "capture", kind=kind, name=f"{label}-{kind}-{sample}", target=target)))
            if box != self.box(field):
                raise RuntimeError("the focused composer moved between samples")
        for i in range(3):
            pair("quiet-" + str(i))
            time.sleep(0.55)
        suffix = " " + TYPING_WORD
        checks[key + "typed"] = (self.type(field, suffix, replace=False, target=target)
                                 and self.value(field) == TYPING_WORD + suffix)
        self.page("await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))")
        pair("typed")
        ink = {}
        for kind, shots in pictures.items():
            facts = evidence[kind] = {"samples": shots}
            if kind == "window" and shots[-1]:
                colors = self.observe(label + " window readability", lambda: window_colors(self.out / shots[-1]["file"]))
                facts["colors"], checks[key + "window_pixels"] = colors, pixels_readable(colors)
            if all(shots):
                result = self.observe(label + " " + kind + " field", lambda: typing_delta(
                    [self.out / s["file"] for s in shots], box, target if kind == "window" else None))
                if result is not None:
                    metric, ink[kind] = result
                    facts.update(metric)
                    if kind == "renderer":
                        checks[key + "typed_pixels"] = metric["visible_word"]
                    elif not metric["visible_word"]:
                        checks[key + "window_typed_pixels"] = False
        if "window" in ink and "renderer" in ink:
            overlap = evidence["changed_ink_overlap"] = echo_overlap(ink["window"], ink["renderer"])
            checks[key + "window_typed_pixels"] = (evidence["window"]["visible_word"]
                                                    and checks[key + "typed_pixels"] and overlap >= 0.60)

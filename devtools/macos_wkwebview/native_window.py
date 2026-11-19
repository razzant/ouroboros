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
OWN_FILES = ("native_window.py", "window_renewal.py", "arm_source.py")
CAPTURE_BRIDGE = HERE.parents[1] / "skills/unix_computer_use/macos_capture.py"
WINDOW_ID = ("shell", "pid", "window", "number")
TYPING_WORD = "Window"
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


def launcher_channel(root, entry, successor, out, tag):
    """Carry the private endpoint across the launcher's own POSIX exec, never changing either tree."""
    hooks = root / "observer"
    hooks.mkdir(exist_ok=True)
    for name in ("native_window.py", "composer_clearance.py"):
        shutil.copyfile(HERE / name, hooks / name)
    shutil.copyfile(CAPTURE_BRIDGE, hooks / "macos_capture.py")
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
                    kind, (0, 0), 0, time.monotonic(), window.windowNumber(), None, char, char, False, 0)
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
                            "frame": json.loads(slots.get("ouro_chat_unconfirmed", "[]")),
                            "contains_fixture_bytes": any(marker in v or encoded in v for v in slots.values())}
            return self.on_main(read)
        raise ValueError("unknown observer operation")


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
            checks[key + "reply_restored"] = self.observe("follow reply in " + prefix, lambda:
                self.follows_quote(reply, prefix, staged.get("quote_text")))
        if staged["filename"] and staged.get("file"):
            notice = not doc["names"] and doc["input_files"] == 0 and "Attach your file again" in doc["notice"]
            checks[key + "reattach_notice"] = notice
            if notice and not project:
                self.attach(staged["filename"], marker, prefix)
                after = self.draft(field)
                checks["attachment_restaged"] = after["names"] == [staged["filename"]] and not after["notice"]

    def shows_project(self, project, target, *, in_list=False, origin=None):
        """Read the named room or its sidebar row in place, without selecting it."""
        if not self.key_window(target):
            raise RuntimeError("the project window was not visible and key")
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
            if (title && visible(title, a.list ? room : title.parentElement)) {
                if (title.textContent === a.name) return a.list || (room.querySelector('#pchat-' + id + '-input') ? true : null);
                if (a.final && title.textContent) return false;
            }
            if (a.list) return false;
            const main = document.querySelector('#chat-input');
            if (a.final && !room && visible(main) && !main.readOnly && !main.disabled &&
                main.dataset.desktopRoomOpening !== '1') return false;
            return null;
        """, id=project["id"], name=project["name"], list=in_list, final=final, origin=origin)
        shown = self.until(read, 35 if in_list else 20)
        if not shown and not in_list:
            shown = read(final=True)  # a visible different room is a finding; an unfinished mount stays unknown
        if not visible_key_view(target, self.state()):
            raise RuntimeError("the active window changed while observing the project")
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

    def custody(self, marker, room):
        slots = self.call("slots", marker=marker, room=room)
        if slots is None:  # older arm has tab storage
            slots = self.page("""
                const get = k => sessionStorage.getItem(k);
                return {text:get('ouro_chat_draft'), reply:JSON.parse(get('ouro_chat_reply') || 'null'),
                    project:get('ouro_chat_draft:'+a.room), frame:JSON.parse(get('ouro_chat_unconfirmed') || '[]')};
            """, room=room)
        return slots

    def bubble(self, msg_id):
        return self.page("""
            const rows = [...document.querySelectorAll('#chat-messages .chat-bubble.user[data-client-message-id]')]
                .filter(row => row.dataset.clientMessageId === a.id);
            const buttons = rows.length === 1 ? rows[0].querySelectorAll('[data-unconfirmed-action=retry]') : [];
            return {count:rows.length, retry:buttons.length === 1 && !buttons[0].disabled &&
                !!buttons[0].getClientRects().length};
        """, id=msg_id)

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


def pixels_readable(shot: dict | None) -> bool | None:
    """Not a uniform dark surface: anti-aliased text and controls leave many colours. People still read the PNG."""
    if not shot or not shot.get("ok"):
        return None
    return shot["distinct_colors"] >= 32 and shot["dark_fraction"] < 0.995


def drawn_rgb(image):
    """Fully transparent pixels have no visible colour, even if their stored RGB channels change."""
    from PIL import Image

    rgba = image.convert("RGBA")
    rgb = Image.new("RGB", rgba.size)
    rgb.paste(rgba, mask=rgba.getchannel("A"))
    return rgb


def typing_delta(paths, box, window=None):
    """Map the CSS field to the WK bitmap or the client area of the shadow-free NSWindow image."""
    from PIL import Image, ImageChops

    left, top, right, bottom, css_w, css_h, start, width, advance = box
    if not (0 <= left < right <= css_w and 0 <= top < bottom <= css_h):
        raise ValueError("the composer was not fully visible")
    frames = []
    for i, path in enumerate(paths):
        with Image.open(path) as image:
            if i == 0:
                size = image.size
                if window:
                    fw, fh = window["frame"][2:]
                    x, y, vw, vh = window["viewport"]
                else:
                    fw, fh = vw, vh = css_w, css_h
                    x = y = 0
                if min(fw, fh, vw, vh) <= 0 or abs(size[0] * fh / fw - size[1]) > 2:
                    raise ValueError("capture aspect ratio disagrees with its native bounds")
                sx, sy = size[0] / fw, size[1] / fh
                scale = sx * vw / css_w
                crop = (round(sx * (x + left * vw / css_w)), round(sy * (y + top * vh / css_h)),
                        round(sx * (x + right * vw / css_w)), round(sy * (y + bottom * vh / css_h)))
                if not (0 <= crop[0] < crop[2] <= size[0] and 0 <= crop[1] < crop[3] <= size[1]):
                    raise ValueError("the field could not be located in this image")
            if image.size != size:
                raise ValueError("the capture dimensions changed around input")
            frames.append(drawn_rgb(image.crop(crop)))
    def difference(a, b):
        return ImageChops.difference(a, b).convert("L").point(lambda level: 255 if level >= 12 else 0)
    quiet = frames[:3]
    noise = max(difference(a, b).histogram()[255] for i, a in enumerate(quiet) for b in quiet[i+1:])
    ink = difference(quiet[0], frames[3])
    for previous in quiet[1:]:
        ink = ImageChops.darker(ink, difference(previous, frames[3]))
    css_size = (max(1, round(right - left)), max(1, round(bottom - top)))
    def css_mask(mask):
        return mask.resize(css_size, Image.Resampling.BOX).point(lambda level: 255 if level >= 32 else 0)
    # The quiet field holds one word. Its already drawn strokes, translated by one measured
    # word and space, predict what the second copy should add in this very capture.
    x, dest, length = (round(v * scale) for v in (start, start + advance, width))
    first = quiet[0].crop((x, 0, x + length, ink.height))
    blank = quiet[0].crop((dest, 0, dest + length, ink.height))
    reference = Image.new("L", ink.size)
    reference.paste(difference(first, blank), (dest, 0))
    changed = css_mask(ink)
    match = echo_overlap(changed, css_mask(reference), tolerance=1)
    count, reference_count = ink.histogram()[255], reference.histogram()[255]
    above_noise = count > 3 * noise + 40 * scale * scale
    same_word = 0.6 * reference_count <= count <= 1.6 * reference_count and match >= 0.70
    return {"noise": noise, "change": count, "scale": scale, "image_size": size, "crop": crop,
            "reference_pixels": reference_count, "word_overlap": match,
            "visible_word": above_noise and same_word}, changed


def echo_overlap(window, renderer, tolerance=2):
    """Fraction of changed ink near the other image's changed ink, measured in both directions in CSS pixels."""
    from PIL import ImageChops, ImageFilter

    def fraction(a, b):
        count = a.histogram()[255]
        near = b.filter(ImageFilter.MaxFilter(2 * tolerance + 1))
        return ImageChops.darker(a, near).histogram()[255] / count if count else 0.0
    return min(fraction(window, renderer), fraction(renderer, window))


def window_colors(path):
    from PIL import Image

    with Image.open(path) as image:
        # Exclude the title bar, border and shadow. Count only central client content.
        crop = drawn_rgb(image.crop((image.width // 12, image.height // 6,
                                     11 * image.width // 12, 11 * image.height // 12)))
        crop.thumbnail((500, 350))
        dark = sum(crop.convert("L").histogram()[:18])
        return {"ok": True, "distinct_colors": len(crop.getcolors(crop.width * crop.height)),
                "dark_fraction": round(dark / (crop.width * crop.height), 5)}

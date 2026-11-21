"""Replace the main Cocoa renderer in its existing pywebview 5.4 BrowserView on AppKit.

The NSWindow, its delegates, its controls and the BrowserView registry keep their identity.
WebKit copies the configuration but shares its content controller, website data store and
scripts. Delayed bridge replies target the view that issued the call. The room session has
its own main-frame reply channel; pywebview never evaluates session data as JavaScript.

A generation is the child, its port and its document key as the lifecycle owner confirmed them.
Whatever this shell itself loads comes from the generation that is this launcher's live child in
the AppKit turn that loads it: a successor's root, the window's first document (bound in that
turn, never when the window was made; without a live child nothing loads until the lifecycle
owner speaks again) and Help's page, kept as a path until the window has settled and never loaded
over unsaved forms. Only WebKit's own word that such a load failed before its view had a document
is asked again, a bounded number of times; a successor that stays silent is refused at its
deadline, as before. A view, document or generation that was overtaken ends only its own part: it
neither discards nor navigates what came later. Every bridge door of the emergency stop takes it
in the turn it arrives, from the installed view's main frame on that root alone: by the document's
ticket or, from a document that got none, by what this shell's own tagged request identified. That older request can deliver a late stop
but never reads drafts.
"""
from __future__ import annotations

import json
import logging
import secrets
from contextlib import ExitStack, contextmanager, suppress
from functools import cache, partial
from types import SimpleNamespace
from threading import Event, RLock, current_thread, main_thread
from time import monotonic
from urllib.parse import parse_qs, urlsplit

from ouroboros import launcher_webkit_diagnostics as diagnostics
from ouroboros.desktop_room_session import (
    ASK, HELD, KEY_ENV, LEGACY_INPUT_PROBE, NOTICES, OLDER_ROOT_PROBE, PANIC_HANDLER, REQUEST_HEADER, WAITS, RoomSession,
    SessionRefused, anchor_navigation, main_frame_request, matches_document, matches_retired_panic, native_notice,
    root_url as _root_url, served_document,
)
from ouroboros.launcher_readiness import absent_rooms

log = logging.getLogger("launcher.webkit")
HANDLER = "ouroborosRoomSession"
_DEFERRED = object()
_EDITORS = {"Files", "Your rules", "Circles"}
_STARTUP_SECONDS = 20  # silence between startup observations, not a total replacement limit
_DRAIN_SECONDS = 0.5
_RETRY_SECONDS = (1, 3)  # the waits before asking again for a load WebKit itself said failed; after them, no more


def _on_appkit(callback, *args, after=None):
    from PyObjCTools import AppHelper
    (AppHelper.callAfter if after is None else partial(AppHelper.callLater, after))(callback, *args)


class _BridgeTarget:
    """pywebview's injection and JS-API result workers capture this view at dispatch."""

    def __init__(self, browser, view):
        self.browser, self.webview = browser, view
        self.lock, self.waiting, self.active = RLock(), set(), True
        self.events = SimpleNamespace(**{name: SimpleNamespace(set=partial(self._event, name))
                                         for name in ("before_load", "loaded", "_pywebviewready")})

    def _event(self, name):
        if name == "before_load":  # synchronous vendor callback on AppKit, already serialized with retirement
            if self.active:
                self.browser.pywebview_window.events.before_load.set()
            return
        with self.lock:
            if self.active:
                getattr(self.browser.pywebview_window.events, name).set()

    def __getattr__(self, name):
        return getattr(self.browser.pywebview_window, name)

    def evaluate(self, script, parse_json=False):
        # 5.4 dispatches evaluateJavaScript on AppKit, then waits on the calling worker.
        # Retiring the document also releases its waiters, including inject_pywebview's expose lock.
        done, result = Event(), []
        with self.lock:
            if not self.active:
                return None
            self.waiting.add(done)

        def completed(value, error):
            result.append(value)
            done.set()

        def evaluate():
            if self.active:
                try:
                    self.webview.evaluateJavaScript_completionHandler_(script, completed)
                except Exception:
                    done.set()
            else:
                done.set()

        try:
            _on_appkit(evaluate)
            done.wait()
        finally:
            with self.lock:
                self.waiting.discard(done)
        value = result[0] if result and self.active else None
        if value and parse_json:
            try:
                return json.loads(value)
            except (TypeError, ValueError):
                pass  # same raw-result fallback as Cocoa; never log the payload
        return value

    run_js = evaluate
    evaluate_js = evaluate  # vendor JS-API result assignment

    def retire(self):
        with self.lock:
            self.active = False
            self.webview = None
            for done in self.waiting:
                done.set()


@cache
def _native_classes():
    import objc
    import WebKit
    from Foundation import NSObject

    class OuroborosMainNavigation(NSObject, protocols=[objc.protocolNamed("WKNavigationDelegate")]):
        def respondsToSelector_(self, selector):
            previous = getattr(self, "previous", None)
            return bool(objc.super(OuroborosMainNavigation, self).respondsToSelector_(selector)
                        or (previous is not None and previous.respondsToSelector_(selector)))

        def forwardingTargetForSelector_(self, selector):
            if self.previous.respondsToSelector_(selector):
                return self.previous
            return objc.super(OuroborosMainNavigation, self).forwardingTargetForSelector_(selector)

        def webView_decidePolicyForNavigationAction_decisionHandler_(self, view, action, handler):
            frame, request = action.targetFrame(), action.request()
            original = self.previous.webView_decidePolicyForNavigationAction_decisionHandler_
            if not self.owner.owns(view) or (frame is None and view is self.owner.candidate):
                self.cancel_action(handler)
                return
            if frame is None or not frame.isMainFrame():
                original(view, action, handler)
                return
            self.history_request = None
            url, kind = request.URL().absoluteString(), action.navigationType()
            tag = request.valueForHTTPHeaderField_(REQUEST_HEADER)
            own = bool(self.challenge and self.issued and tag == self.challenge)  # this shell's request on its way
            if view is self.owner.view and self.owner.first and not own:
                # A window without a document takes only this shell's own request for its live child's root. Whatever
                # else is asked for, pywebview's address from when the window was made included (its child may be
                # gone), is cancelled and answered by that request, unless that one is already on its way.
                self.cancel_action(handler)
                self.owner._first_document()
                return
            if request.HTTPMethod() != "GET" or not _root_url(url, self.port) or action.shouldPerformDownload():
                if view is self.owner.candidate:
                    self.cancel_action(handler)
                    self.owner.refuse("successor left the selected main address")
                    return
                original(view, action, handler)
                return
            if kind == WebKit.WKNavigationTypeBackForward:
                # Let WebKit traverse the original history item (including SPA layer sentinels).
                # Only a cross-document traversal commits and needs a newly verified response.
                self.history_request = request
                original(view, action, handler)
                return
            here = frame.request().URL()
            if anchor_navigation(here.absoluteString() if here else None, url, kind == WebKit.WKNavigationTypeReload):
                original(view, action, handler)
                return
            if view is self.owner.view and not self.owner.first:  # its page's own reloads; a first document has no page
                refresh = (here and str(here.absoluteString()) != str(url)
                           and "_ouro_refresh" in parse_qs(urlsplit(str(url)).query))
                if self.owner.requested or (refresh and self.owner.pending):
                    self.cancel_action(handler)
                    return
                live = self.owner.live()
                if refresh and live and self.generation is not live and self.owner.reconnected is not live:
                    # Let ordinary socket recovery retry a failed replacement at its confirmed port.
                    # Navigating the usable view directly would commit even an error page there.
                    self.cancel_action(handler)
                    self.owner.pending, self.owner.requested = live, True
                    self.owner._post(live)
                    return
            if own:
                self.issued = False
                original(view, action, handler)
                return
            self.cancel_action(handler)
            if not (tag and self.issued):
                # Not an earlier tagged request the one on its way overtook: tagging that again would void the
                # challenge of the later one, whose own turn here would then tag a third, without end.
                self.expected_navigation = None  # ignore cancellation of the untagged request
                self.load_tagged(view, request)

        @objc.python_method
        def load_tagged(self, view, request):
            # A page that naturally reloads after a server crash must use that server's key.
            # The crash path itself neither navigates nor creates a view.
            live = self.owner.live()
            if view is self.owner.view and live and self.port == live[1]:
                self.key = live[2]
            self.challenge, self.issued, self.next_ticket = secrets.token_hex(32), True, ""
            tagged = request.mutableCopy()
            tagged.setValue_forHTTPHeaderField_(self.challenge, REQUEST_HEADER)
            self.expected_navigation = view.loadRequest_(tagged)
            return self.expected_navigation

        @objc.python_method
        def cancel_action(self, handler):
            # pywebview 5.4 supplies the missing Cocoa block signature on this same delegate path.
            if not handler.__block_signature__:
                from webview.platforms.cocoa import BrowserView
                handler.__block_signature__ = BrowserView.pyobjc_method_signature(b"v@i")
            handler(WebKit.WKNavigationActionPolicyCancel)

        def webView_decidePolicyForNavigationResponse_decisionHandler_(self, view, response, handler):
            if not self.owner.owns(view):
                self.cancel_action(handler)  # both policy enums use Cancel (0)
                return
            if response.isForMainFrame():
                http, live = response.response(), self.owner.live()
                # An unstamped root is an older server's. In the view on screen it is this installation's only while
                # this launcher's own live child holds that port; a successor's is confirmed separately before it shows.
                self.next_source, self.next_ticket = served_document(
                    http, response, self.port, self.key, self.challenge,
                    allow_unmarked=self is self.owner.candidate_nav or bool(live and live[1] == self.port))
                if self.next_source == "unmarked":
                    self.response_status = http.statusCode()
                self.challenge = ""
                if view is self.owner.candidate and not self.next_source:
                    self.cancel_action(handler)
                    self.owner.refuse("successor did not verify its selected main document")
                    return
            self.previous.webView_decidePolicyForNavigationResponse_decisionHandler_(view, response, handler)

        def webView_didCommitNavigation_(self, view, navigation):
            if not self.owner.owns(view):
                return
            if view is self.owner.candidate and self.committed_navigation is not None:
                self.owner.refuse("successor changed document during startup")
                return
            if view is self.owner.view and view is self.owner.browser.webview:
                self.owner.bind_bridge(view, leaving=self)
            self.committed_navigation = navigation
            self.ticket = self.next_ticket if navigation is self.expected_navigation else ""
            self.source = self.next_source if navigation is self.expected_navigation else ""
            self.document_key = self.key if self.ticket else ""
            self.next_source = self.next_ticket = ""
            if view is self.owner.view:
                self.room_project = None  # the next document has yet to open its room
                if self.legacy != "separate":
                    self.legacy = None
                latest = self.owner.latest
                if self.ticket and latest and (self.port, self.key) == latest[1:]:
                    self.generation = latest
                if not self.ticket and self.owner.session.has_room_input:
                    log.info("WebKit room input retained in shell; this document is not verified")
                self.owner.document_left()
                if not self.owner.owns(view):
                    return  # its exchange ended here and retired this view
            if self.owner.candidate is view and not self.source:
                self.owner.refuse("successor changed document during startup")
                return
            if self.previous.respondsToSelector_("webView:didCommitNavigation:"):
                self.previous.webView_didCommitNavigation_(view, navigation)
            history, self.history_request = self.history_request, None
            if history is not None and view is self.owner.view and view.URL() == history.URL():
                self.committed_navigation = None
                self.load_tagged(view, history)
            elif view is self.owner.view:
                self.owner._help()  # Help's page, if it waited for this document

        def webView_didFinishNavigation_(self, view, navigation):
            if navigation is not self.committed_navigation:
                return
            if self.owner.candidate is view and self.finished is not navigation:
                self.finished, self.startup_progress = navigation, monotonic()
                self.owner.present_if_ready()
            elif self.owner.view is view:
                self.previous.webView_didFinishNavigation_(view, navigation)

        def webView_didFailProvisionalNavigation_withError_(self, view, navigation, error):
            self.navigation_failed("webView:didFailProvisionalNavigation:withError:", view, navigation, error)

        def webView_didFailNavigation_withError_(self, view, navigation, error):
            self.navigation_failed("webView:didFailNavigation:withError:", view, navigation, error)

        @objc.python_method
        def navigation_failed(self, selector, view, navigation, error):
            owner = self.owner
            if navigation is self.expected_navigation and (view is owner.candidate or (view is owner.view and owner.first)):
                owner.navigation_failed(view, self, "Provisional" in selector)
            if view is owner.view and self.previous.respondsToSelector_(selector):
                getattr(self.previous, selector.replace(":", "_"))(view, navigation, error)

    class OuroborosRoomMessages(NSObject, protocols=[objc.protocolNamed("WKScriptMessageHandlerWithReply")]):
        def userContentController_didReceiveScriptMessage_replyHandler_(self, controller, message, reply):
            try:
                receiver = self.owner.receive_panic if message.name() == PANIC_HANDLER else self.owner.receive
                answer = receiver(message, reply)
            except Exception:
                reply(None, "Desktop room session is unavailable for this document.")
            else:
                if answer is not _DEFERRED:
                    reply(answer, None)

    class OuroborosCurrentViewMessages(NSObject):
        def userContentController_didReceiveScriptMessage_(self, controller, message):
            if message.webView() is not self.owner.browser.webview:
                if self.is_bridge:
                    self.owner.retired_panic(message)
                return
            if self.is_bridge:
                body = json.loads(message.body())
                name = body.get("funcName")
                if not isinstance(name, str) or any(not part or part.startswith("_") for part in name.split(".")):
                    return  # pywebview publishes public API methods; private native owners are never a JS API
                if name == "panic_stop":
                    # With or without a ticket, a stop is verified in the turn it arrives and is the launcher's at
                    # once: no worker of the vendor bridge stands between. Its call below only answers the page.
                    if not self.owner.current_panic(message, body.get("params")):
                        self.owner.retired_panic(message)
                        return
                    self.owner.background.request_panic("desktop_bridge")
                if name == "pywebviewAsyncCallback" and body.get("id") not in self.owner.window._callbacks:
                    return  # a promise from a retired document has no live callback in the wrapper
            self.original.userContentController_didReceiveScriptMessage_(controller, message)

    return OuroborosMainNavigation, OuroborosRoomMessages, OuroborosCurrentViewMessages


class MainRenderer:
    def __init__(self, root, current):
        self.session = RoomSession.receive(root)
        self.current = current  # reads this launcher's confirmed process/port and shutdown latch
        self.key = ""
        self.window = self.view = self.navigation = self.observation = None
        self.pending = self.presented = self.latest = self.reconnected = None
        self.candidate = self.candidate_nav = self.candidate_observation = None
        self.requested = self.installed = False
        self.close_observer = None
        self.bridge = None
        self.previous_panic = self.exchange = self.source_drain = None
        self.resumes = []  # resume_queue answers owed at the lifecycle owner's next word
        self.waiting_for_builtin = False
        self.release_vendor_retain = True
        self.first = False  # True until the window's first document commits: only its live child's root may be it
        self.help = None  # the path of Help's page, until the view the window settles on has loaded it
        self.failures = (None, 0)  # a generation, and how often WebKit said a load of this shell for it failed

    def server_environment(self, env):
        self.key = secrets.token_hex(32)
        env[KEY_ENV] = self.key

    def live(self):
        """The confirmed generation while it is this launcher's own live child; None between servers."""
        latest = self.latest
        return latest if latest is not None and self.current(*latest[:2]) else None

    def attach(self, window, background):
        self.window, self.background = window, background
        window.events.before_show += self._before_show

    def _before_show(self):
        if not getattr(self.window.gui, "__name__", "").endswith(".cocoa"):
            return
        from webview.platforms.cocoa import BrowserView

        self.browser = BrowserView.instances[self.window.uid]
        self.view = self.browser.webview
        if (live := self.live()) is not None:
            # pywebview loads the window's address next in this turn: the child confirmed now, which need not be the
            # one confirmed when the window was made. With the delegate below installed that request is only a cue.
            self.window.real_url = f"http://127.0.0.1:{live[1]}/"
        try:
            import WebKit
            from AppKit import NSWindowWillCloseNotification
            from Foundation import NSNotificationCenter

            self.close_observer = NSNotificationCenter.defaultCenter().addObserverForName_object_queue_usingBlock_(
                NSWindowWillCloseNotification, self.browser.window, None, lambda note: self._dispose())
            _, messages, current_messages = _native_classes()
            controller = self.view.configuration().userContentController()
            receiver = messages.alloc().init()
            receiver.owner = self
            controller.addScriptMessageHandlerWithReply_contentWorld_name_(
                receiver, WebKit.WKContentWorld.pageWorld(), HANDLER)
            for name, original in (("jsBridge", self.browser.js_bridge), ("browserDelegate", self.browser._browserDelegate)):
                guarded = current_messages.alloc().init()
                guarded.owner, guarded.original, guarded.is_bridge = self, original, name == "jsBridge"
                controller.removeScriptMessageHandlerForName_(name)
                try:
                    controller.addScriptMessageHandler_name_(guarded, name)
                except Exception:
                    controller.addScriptMessageHandler_name_(original, name)
                    raise
            self.navigation, self.first = self._navigation(self.view, None), True
            self.bind_bridge(self.view)
            self.installed = True
            try:
                controller.addScriptMessageHandlerWithReply_contentWorld_name_(
                    receiver, WebKit.WKContentWorld.pageWorld(), PANIC_HANDLER)
            except Exception as exc:
                log.warning("Early WebKit Panic channel unavailable (%s)", type(exc).__name__)
        except Exception as exc:
            log.warning("WebKit room session registration failed (%s); keeping this document", type(exc).__name__)
        self.observation = diagnostics.attach(self.view)
        if self.pending:
            # In a later turn. pywebview loads the window's address next in this one, and WebKit keeps only the last
            # request it was given: a request of this shell made here would be dropped for that one.
            self._post(self.pending)

    def _first_document(self):
        """The window's first document is only the root of the generation that is this launcher's live child in this
        AppKit turn, asked for by this shell's own tagged request. The child confirmed when the window was made may be
        gone, and without a live child nothing loads: the delegate cancels every other request, and pywebview's own,
        the lifecycle owner's next word or a retry after WebKit's word of failure ask here again. A navigation bound
        to the live child is a request already on its way."""
        nav, live = self.navigation, self.live()
        if not self.first or live is None or nav.generation is live:
            return
        nav.generation, (nav.port, nav.key) = live, live[1:]
        self.presented = live
        self.window.real_url = self.browser.url = url = f"http://127.0.0.1:{live[1]}/"
        from Foundation import NSURL, NSURLRequest
        if nav.load_tagged(self.view, NSURLRequest.requestWithURL_(NSURL.URLWithString_(url))) is None:
            nav.generation = nav.port = None  # WebKit took no request: the document is still owed

    def bind_bridge(self, view, leaving=None):
        if self.bridge is not None:
            if leaving is not None and leaving.ticket:
                import objc
                # A WKScriptMessage can outlive the Python proxy of its originating WKWebView. A document that got no
                # ticket retires nobody: the last verified document's posted stop stays owed across it.
                self.previous_panic = (objc.pyobjc_id(self.bridge.webview), leaving.port, leaving.ticket,
                                       leaving.document_key, self.bridge.webview is view)
            self.bridge.retire()
            self.window._callbacks.clear()
        self.bridge = _BridgeTarget(self.browser, view)
        self.browser.js_bridge.window = self.bridge
        self.browser.evaluate_js = self.bridge.evaluate

    def retired_panic(self, message):
        # A bridge stop posted before its document retired: verified like the reply door's, then straight to the
        # launcher's own stop. Its promise went with its document, so nothing is answered and nothing is retried.
        try:
            body = json.loads(message.body())
        except (ValueError, TypeError):
            return
        if (isinstance(body, dict) and body.get("funcName") == "panic_stop" and isinstance(body.get("id"), str)
                and matches_retired_panic(self.previous_panic, message, body.get("params"))):
            self.previous_panic = None
            self.background.request_panic("desktop_bridge")

    def current_panic(self, message, params):
        """A stop from the installed view's main frame, on the root this shell loaded there from its own server.
        ``params`` is the door's argument list. A stamped document proves itself by its ticket. Only a document that
        got none (an older or a custom interface, whose call carries no argument) is taken by what this shell's own
        tagged request identified; a subframe, another address or an unidentified document never is."""
        view = message.webView()
        nav = self.candidate_nav if view is self.candidate else self.navigation
        if (view is None or view is not self.browser.webview or not self.owns(view) or not nav
                or not main_frame_request(message, nav.port)):
            return False
        if params == []:
            return bool(nav.source) and not nav.ticket
        return isinstance(params, list) and len(params) == 1 and matches_document(params[0], nav.ticket)

    def receive_panic(self, message, reply):
        ticket = message.body()
        if not self.current_panic(message, [ticket]):
            if not matches_retired_panic(self.previous_panic, message, [ticket]):
                raise ValueError("unverified document")
            self.previous_panic = None
        self.background.request_panic("desktop_bridge")
        return {"status": "stopping"}

    @contextmanager
    def prepare_exec(self, probed_builtin, *, transfer):
        with ExitStack() as cleanup:
            nav = self.navigation
            ticket = nav.ticket if nav else ""
            changed = Event()
            if ticket and nav.legacy != "separate":
                # The lifecycle worker may wait once. AppKit and the existing Panic door
                # stay live. On refusal the same shell starts the server and replaces its view.
                if current_thread() is main_thread():
                    raise SessionRefused("source handoff cannot wait on AppKit")
                import AppKit
                drain = SimpleNamespace(outgoing=nav, token=secrets.token_hex(16), done=Event(),
                                        closed=Event(), confirmed=False, monitor=None)

                def input_arrived(event):
                    changed.set()
                    return event

                def start():
                    if drain.closed.is_set() or self.navigation is not nav or nav.ticket != ticket:
                        return
                    try:
                        # A press already handed to WebKit can precede its storage message.
                        mask = (AppKit.NSEventMaskKeyDown | AppKit.NSEventMaskKeyUp
                                | AppKit.NSEventMaskLeftMouseDown | AppKit.NSEventMaskLeftMouseUp
                                | AppKit.NSEventMaskRightMouseDown | AppKit.NSEventMaskRightMouseUp
                                | AppKit.NSEventMaskOtherMouseDown | AppKit.NSEventMaskOtherMouseUp)
                        drain.monitor = AppKit.NSEvent.addLocalMonitorForEventsMatchingMask_handler_(mask, input_arrived)
                        if drain.monitor is not None:
                            self.source_drain = drain
                            self._ask_for_final_input(self.view, drain)
                    except Exception:
                        drain.done.set()

                def end():
                    if self.source_drain is drain:
                        self.source_drain = None
                    if drain.monitor is not None:
                        AppKit.NSEvent.removeMonitor_(drain.monitor)

                cleanup.callback(_on_appkit, end)
                cleanup.callback(drain.closed.set)
                _on_appkit(start)
                if not drain.done.wait(_DRAIN_SECONDS) or not drain.confirmed:
                    raise SessionRefused("previous window did not confirm its final room input")

            validate_slots, inherit = cleanup.enter_context(self.session.prepare_exec(transfer=transfer))
            if self.session.needs_transfer and not transfer:
                raise SessionRefused("room data arrived after the source capability probe")
            if self.session.has_room_input and not probed_builtin:
                raise SessionRefused("room input arrived after the source capability probe")

            def validate():
                if changed.is_set() or self.navigation is not nav or (nav and nav.ticket != ticket):
                    raise SessionRefused("input continued during the source handoff")
                validate_slots()

            @contextmanager
            def pass_to_exec(env):
                validate()
                with inherit(env):
                    yield
            yield validate, pass_to_exec

    def _navigation(self, view, generation):
        cls, _, _ = _native_classes()
        nav = cls.alloc().init()
        nav.owner, nav.previous = self, self.browser._browserDelegate
        nav.generation, nav.issued, nav.startup_progress = generation, False, monotonic()
        nav.finished = nav.expected_navigation = nav.committed_navigation = nav.history_request = None
        nav.port, nav.key = generation[1:] if generation else (None, "")  # a first document is bound when it loads
        nav.document_key = nav.key
        nav.ticket = nav.next_source = nav.source = nav.next_ticket = nav.challenge = ""
        nav.ready_reply = nav.room_project = nav.legacy = None
        view.setNavigationDelegate_(nav)
        return nav

    def owns(self, view):
        return view is self.view or view is self.candidate

    def receive(self, message, reply=None):
        view, body = message.webView(), message.body()
        nav = self.candidate_nav if view is self.candidate else self.navigation
        if view is None or not self.owns(view) or nav is None or not main_frame_request(message, nav.port):
            raise ValueError("unverified frame")
        if not matches_document(body["ticket"], nav.ticket):
            raise ValueError("unverified document")
        op = body["op"]
        if op == "open":
            if self.exchange and self.exchange.shown and view is self.view:
                raise ValueError("outgoing room can only finish its handoff")
            if view is self.candidate:
                if not self._current(nav.generation):
                    self.retire_generation(nav.generation)
                    raise ValueError("replacement superseded")
                if not body.get("staged"):
                    # Protocol 1 opens before mounting and expects a flat dictionary.
                    # Stage its verified public page without seeding it with private slots.
                    if nav.legacy or reply is None:
                        raise ValueError("unexpected room open")
                    nav.legacy, nav.ready_reply = "opening", reply
                    self.present_if_ready()
                    return _DEFERRED
                if nav.room_project is not None:
                    raise ValueError("room already opening")
                nav.room_project = self.session.items.get("ouro_chat_active_project", "")
                nav.startup_progress = monotonic()
                return {"mountProject": nav.room_project}
            if self.session.restore_failed:
                self.session.restore_failed = False
                self.notify_previous("ouro:room-restore-failed")
            if body.get("staged"):
                nav.legacy = None
                nav.room_project = self.session.items.get("ouro_chat_active_project", "")
            return {} if nav.legacy else self.session.items.copy()
        if view is self.candidate and op == "project_importing":
            if (not nav.room_project or nav.ready_reply is not None or body.get("project") != nav.room_project
                    or body.get("stage", "boot") not in ("mount", "retry", "migrating", "boot") or not self._can_present(nav)):
                raise ValueError("obsolete project read")
            nav.startup_progress = monotonic()
            return True
        if view is self.candidate and op == "ready":
            # A neutral selection shell still acknowledges the old address; it owns none of that room's edits.
            if (reply is None or nav.ready_reply or nav.room_project is None or nav.room_project != body.get("project")
                    or (body.get("selection") and not nav.room_project)):
                raise ValueError("unexpected room")
            nav.ready_reply, nav.startup_progress = reply, monotonic()
            self.present_if_ready()
            return _DEFERRED
        if view is not self.view:
            raise ValueError("inactive renderer")
        if nav.legacy == "separate" and op == "update":
            return True  # this older page owns its local input; the previous room stays held in the shell
        if op == "drained":
            exchange = self.exchange or self.source_drain
            if (exchange is None or exchange.token != body.get("handoff") or nav is not exchange.outgoing
                    or (exchange is self.source_drain and exchange.closed.is_set())):
                raise ValueError("expired handoff")
            items, editors = json.loads(body["items"]), set(body["editors"])
            if not isinstance(items, dict) or not editors <= _EDITORS:
                raise ValueError("invalid room handoff")
            self.session.apply([(key, None) for key in self.session.items.keys() - items.keys()] + list(items.items()))
            self.session.edits = editors
            if exchange is self.source_drain:
                exchange.confirmed = True
                exchange.done.set()
            else:
                self.finish_exchange(exchange, confirmed=True)
            return True
        if op == "resume_queue":
            if (answer := self._continues(nav)) is None and reply is not None:
                self.resumes.append((nav, reply))  # no timer and no guess: the lifecycle owner's next word answers
                return _DEFERRED
            return bool(answer)
        if op == "update":
            self.session.apply(json.loads(body["changes"]))
            if self.waiting_for_builtin and self.pending and not self.session.has_room_input:
                self.waiting_for_builtin = False
                self._post(self.pending)
        elif op == "editor" and body["name"] in _EDITORS:
            if body["dirty"]:
                self.session.edits.add(body["name"])
            else:
                self.session.edits.discard(body["name"])
                if self.pending and not self.session.edits:
                    self._post(self.pending)
                else:
                    self._help()
        else:
            raise ValueError("unknown room operation")
        return True

    def _continues(self, nav):
        # May this page's queued sends go on: only on the ordinary continuation of its own server at its port, with no
        # replacement asked. None: that server is gone and its successor unconfirmed, as when a crash reconnect outruns
        # the launcher's probe; refusing there would turn the queue into held sends.
        latest = self.latest
        if (self.requested or latest is None or nav.port != latest[1]
                or not (nav.generation is latest or self.reconnected is latest)):
            return False
        return True if self.current(*latest[:2]) else None

    def _answer_resumes(self):
        waiting, self.resumes = self.resumes, []
        for nav, reply in waiting:
            if (answer := self._continues(nav)) is None:
                self.resumes.append((nav, reply))  # still between servers
            else:
                reply(answer, None)

    def _word(self):
        # AppKit's turn after the lifecycle owner spoke: waiting pages are answered, a window still without its first
        # document takes it from the child confirmed now, and Help's page looks for its view and endpoint again.
        self._answer_resumes()
        self._first_document()
        self._help()

    def replacement_requested(self):
        self.requested = True
        if self.observation is not None:
            self.observation.replacement_requested()
        with suppress(Exception):
            _on_appkit(self._word)

    def confirmed(self, proc, port, *, replace):
        previous = self.latest
        generation = self.latest = (proc, port, self.key)
        continued = (previous is self.presented or previous is self.reconnected
                     or (self.navigation is not None and previous is self.navigation.generation))
        self.reconnected = generation if not replace and continued else None
        if replace:
            self.pending, self.requested = generation, True
            # Its registry has settled: a room it positively no longer has stops holding the interface choice. The
            # answer counts only if the child that gave it is still the confirmed live one once it is read.
            gone = absent_rooms(port, self.session.rooms)
            if self.current(proc, port):
                self.session.gone |= gone
        with suppress(Exception):  # without AppKit there is no page whose answer waits
            _on_appkit(self._word)

    def server_exited(self, proc):
        # Called by the lifecycle owner, before its next start. Dispose on AppKit by
        # generation: an already queued callback may run after a newer successor arrived.
        generation = self.pending
        if generation is not None and generation[0] is not proc:
            return False
        self.pending, self.requested = None, False
        if generation is None:
            return False
        self._post(generation)
        return True

    def schedule_confirmed(self):
        if self.view is not None and self.pending is not None:
            self._post(self.pending)

    def load(self, url):
        # Help's page, from pywebview's menu thread, as the address read there. Only its path is the action: the
        # child that address named may be gone before AppKit loads anything.
        target = urlsplit(url)
        if target.path.startswith("/"):
            _on_appkit(self._help, target.path + ("?" + target.query if target.query else ""))

    def _help(self, path=None):
        """Help's page, on AppKit: loaded once, in the view the window has settled on, from the generation that is
        this launcher's live child in this turn. Until then its path is carried. A replacement that is asked for
        settles first, from its request until its successor is installed or given up: a view loaded now would lose
        the page to that successor, and a shown successor refuses any address but its root. A document with unsaved
        forms is never navigated away for it: the shell says so when Help is chosen, and the page follows the owner's
        last edit (and the fresh window those edits held)."""
        if path is not None:
            self.help = path
        live = self.live()
        if self.help is None or self.view is None or live is None or self.first or self.requested:
            return
        if self.session.edits:
            if path is not None:
                self.notify_previous(WAITS, ", ".join(sorted(self.session.edits)))  # fixed names only
            return
        if self.pending is not None and not self.waiting_for_builtin:
            return  # a held replacement is being asked again; one waiting for room input sends its owner to Help
        path, self.help = self.help, None
        from Foundation import NSURL, NSURLRequest
        self.view.loadRequest_(NSURLRequest.requestWithURL_(NSURL.URLWithString_(f"http://127.0.0.1:{live[1]}{path}")))

    def document_left(self):
        """The view on screen committed another document. The one that left owned this window's unsaved forms and
        its last room input: it can confirm neither now, and it never vetoes what came after it. An exchange it was
        part of ends without its word, keeping the successor; a replacement its forms held is asked again."""
        exchange, held = self.exchange, bool(self.session.edits)
        self.first = False
        self.session.edits.clear()
        if exchange is not None:
            self.finish_exchange(exchange, confirmed=False)
        elif held and self.pending:
            self._post(self.pending)

    def navigation_failed(self, view, nav, provisional):
        """WebKit's own word that a load this shell issued failed before its view had a document. While that generation
        is still this launcher's live child a provisional failure is asked again after a pause, a bounded number of
        times: nothing else would ask, since an old page whose server moved to another port never can. Past that
        bound, or once its document had committed, a successor is refused as before. A child that is gone is the
        lifecycle owner's to follow, so what it still owes for that child stays as it is. No probe or timer stands in
        for WebKit's word."""
        generation = nav.generation
        count = self.failures[1] + 1 if self.failures[0] is generation else 1
        self.failures = generation, count
        live = generation is not None and self.live() is generation
        wait = _RETRY_SECONDS[count - 1] if live and provisional and count <= len(_RETRY_SECONDS) else None
        log.warning("WebKit navigation failed before its view had a document (%s)",
                    "asking again" if wait is not None else "its child is gone" if not live else "no further attempt")
        if view is self.view:
            nav.generation = nav.port = None  # owed again: the retry, or the lifecycle owner's next word, loads it
            if wait is not None:
                _on_appkit(self._first_document, after=wait)
        elif live and wait is None:
            self.refuse("successor navigation failed before the room opened")
        else:
            self.drop_candidate()
            if wait is not None and self.pending is generation:
                _on_appkit(self._replace, generation, after=wait)

    def _post(self, generation):
        try:
            _on_appkit(self._replace, generation)
        except Exception as exc:
            if self.pending is generation:
                self.pending, self.requested = None, False
            log.warning("WebKit replacement could not be scheduled (%s)", type(exc).__name__)

    def _current(self, generation):
        return (self.view is not None and self.pending is generation and self.current(*generation[:2])
                and self.browser.webview is (self.candidate if self.exchange and self.exchange.shown else self.view))

    def _replace(self, generation):
        if not self._current(generation):
            self.retire_generation(generation)
            return
        if self.first and self.presented != generation:
            self._first_document()  # a window still without its first document takes it from this child instead
        if self.presented == generation:
            self.pending, self.requested = None, False
            self._help()
            return
        if self.candidate_nav is not None and self.candidate_nav.generation is generation:
            return  # another AppKit turn already started this successor
        self.requested = True  # before an earlier candidate goes: Help's page waits for this successor
        self.drop_candidate()
        self.waiting_for_builtin = False
        if not self.installed:
            self.refuse("session registration unavailable")
            return
        old = self.view
        try:
            from Foundation import NSURL, NSURLRequest
            from webview.platforms.cocoa import BrowserView

            view = BrowserView.WebKitHost.alloc().initWithFrame_configuration_(old.frame(), old.configuration())
            view.pywebview_window = self.window
            view.setAutoresizingMask_(old.autoresizingMask())
            view.setCustomUserAgent_(old.customUserAgent())
            view.setUIDelegate_(old.UIDelegate())
            view.setAllowsBackForwardNavigationGestures_(old.allowsBackForwardNavigationGestures())
            view.setAllowsMagnification_(old.allowsMagnification())
            view.setMagnification_(old.magnification())  # AppKit pinch zoom is separate from the app's pageZoom
            self.candidate = view
            self.candidate_nav = self._navigation(view, generation)
            self.candidate_observation = diagnostics.attach(view)
            self.background.chrome.prepare_renderer(view)
            _on_appkit(self.startup_expired, self.candidate_nav, after=_STARTUP_SECONDS)
            url = NSURL.URLWithString_(f"http://127.0.0.1:{generation[1]}/")
            if self.candidate_nav.load_tagged(view, NSURLRequest.requestWithURL_(url)) is None:
                raise RuntimeError("navigation refused")
            log.info("WebKit successor navigation requested")
        except Exception as exc:
            self.refuse("successor navigation could not start (%s)" % type(exc).__name__)

    def startup_expired(self, nav):
        if self.candidate_nav is nav and self.exchange is None:
            if (remaining := nav.startup_progress + _STARTUP_SECONDS - monotonic()) > 0:
                _on_appkit(self.startup_expired, nav, after=remaining)
            else:
                self.refuse("timed out waiting for the successor window to finish startup")

    def held_editor(self, asking=None):
        names = ", ".join(sorted(self.session.edits))  # fixed names only
        log.info("WebKit replacement held for unsaved edits in %s", names)
        if asking is None:
            self.requested = False
            self.notify_previous(HELD, names)
        elif not self.notify_previous(ASK, names, partial(self.edits_decided, asking)):
            self.edits_decided(asking, False)  # no sheet, no answer: nothing is given up

    def edits_decided(self, exchange, fresh):
        if self.exchange is exchange:
            if fresh:  # the owner's own word: this exchange installs the fresh view without those edits
                self.session.edits.clear()
                self.finish_exchange(exchange, confirmed=False)
            else:  # back to the edits; finishing them asks for the fresh window again, at once if they already are
                self.drop_candidate()
                self.requested = False
                if self.pending and not self.session.edits:
                    self._post(self.pending)

    def notify_previous(self, event, editors="", decide=None):
        if self.view is not None:
            try:
                # An older, custom or absent interface has no listener for these notices; a holding window may not paint.
                if event in NOTICES and (event in (HELD, ASK, WAITS) or not self.navigation
                                         or self.navigation.room_project is None):
                    native_notice(self.browser.window, event, editors, decide)
                    if event != HELD:  # a holding page that still paints keeps its own reminder as well
                        return True
                # Only a fixed event name goes to the page. Recovery never waits for its response.
                self.view.evaluateJavaScript_completionHandler_(f"window.dispatchEvent(new Event('{event}'))", None)
            except Exception as exc:
                log.info("Previous WebKit view could not display the notice (%s)", type(exc).__name__)
        return False

    def _can_present(self, nav):
        if not self._current(nav.generation):
            self.retire_generation(nav.generation)
        elif nav.ticket and not nav.legacy and nav.room_project != self.session.items.get("ouro_chat_active_project", ""):
            # The owner switched rooms while the candidate was mounting. Let the registry
            # resolve the new selection rather than presenting another room's input.
            self.drop_candidate()
            self._post(nav.generation)
        else:
            return True
        return False

    def present_if_ready(self):
        nav = self.candidate_nav
        if (nav is None or self.exchange is not None or nav.finished is None
                or nav.finished is not nav.expected_navigation or not nav.source):
            return
        if nav.source == "unmarked":
            if nav.legacy is None:
                self._identify_older_root(self.candidate, nav)
            return
        if nav.legacy and nav.legacy != "ready":
            if nav.legacy == "opening":
                self._check_legacy_input(self.candidate, nav)
            return
        if (nav.ticket and nav.ready_reply is None) or not self._can_present(nav):
            return
        old, outgoing = self.view, self.navigation
        collecting = outgoing is not None and bool(outgoing.ticket) and outgoing.legacy != "separate"
        exchange = self.exchange = SimpleNamespace(outgoing=outgoing, token=secrets.token_hex(16), shown=False, asked=False)
        # Keep an editable previous room on screen while it can help the owner save input a different interface
        # cannot take. A silent page cannot veto recovery; over its known unsaved forms the owner decides.
        hold = collecting and (self.session.edits or ((not nav.ticket or nav.legacy) and self.session.has_room_input))
        if not hold and not self._show_candidate(exchange):
            return
        if not collecting:
            self.finish_exchange(exchange, confirmed=bool(outgoing and outgoing.source in ("directory", "none")))
            return
        try:
            _on_appkit(self.finish_exchange, exchange, False, after=_DRAIN_SECONDS)
            self._ask_for_final_input(old, exchange)
        except Exception:
            self.finish_exchange(exchange, confirmed=False)

    @staticmethod
    def _ask_for_final_input(view, exchange):
        view.evaluateJavaScript_completionHandler_(
            f"window.dispatchEvent(new CustomEvent('ouro:room-handoff', {{detail:'{exchange.token}'}}))", None)

    def _follow(self, view):
        # Zoom, title bar and appearance belong to the window's content view: they follow a view as it is presented
        # and return with the previous one when a replacement is reversed. A failing control vetoes neither.
        for control in (self.background.zoom, self.background.chrome):
            try:
                control.renderer_changed(view)
            except Exception as exc:
                log.warning("WebKit window control could not follow the installed view (%s)", type(exc).__name__)

    def _show_candidate(self, exchange):
        old, new, nav, native = self.view, self.candidate, self.candidate_nav, self.browser.window
        try:
            responder = native.firstResponder()
            exchange.focus = responder is old or (hasattr(responder, "isDescendantOf_") and responder.isDescendantOf_(old))
            events = (self.window.events.loaded, self.window.events.before_load, self.window.events._pywebviewready)
            exchange.events = [(event, event.is_set()) for event in events]
            # Transfer input to the initialized field first. The detached document can
            # finish its last input event but cannot receive the owner's next keystroke.
            native.setContentView_(new)
            exchange.shown = True
            self.browser.webview = new
            self.bind_bridge(new, leaving=self.navigation)
            self._follow(new)  # before the vendor's didFinish below fires before_load, which reads the declaration
            for event in events:
                event.clear()
            if exchange.focus:
                native.makeFirstResponder_(new)
            nav.previous.webView_didFinishNavigation_(new, nav.finished)
            return True
        except Exception as exc:
            self.refuse(f"native window could not present the successor ({type(exc).__name__})")
            return False

    def _check_legacy_input(self, view, nav):
        nav.legacy = "checking"

        def checked(ready, error):
            if nav is self.candidate_nav and nav.finished is nav.committed_navigation:
                if error or not ready:
                    self.refuse("previous interface did not initialize its room input")
                else:
                    nav.legacy = "ready"
                    self.present_if_ready()
        try:
            import WebKit
            view.callAsyncJavaScript_arguments_inFrame_inContentWorld_completionHandler_(
                LEGACY_INPUT_PROBE, {}, None, WebKit.WKContentWorld.pageWorld(), checked)
        except Exception:
            checked(False, True)

    def _identify_older_root(self, view, nav):
        nav.legacy = "checking"

        def identified(source, error):
            if nav is not self.candidate_nav or nav.finished is not nav.committed_navigation:
                return
            if (error or source not in ("builtin", "directory", "none")
                    or nav.response_status != (404 if source == "none" else 200)):
                self.refuse("older server did not confirm its selected main interface")
                return
            nav.source, nav.legacy = source, "opening" if source == "builtin" else None
            self.present_if_ready()
        try:
            import WebKit
            view.callAsyncJavaScript_arguments_inFrame_inContentWorld_completionHandler_(
                OLDER_ROOT_PROBE, {"stateURL": f"http://127.0.0.1:{nav.port}/api/state"}, None,
                WebKit.WKContentWorld.defaultClientWorld(), identified)
        except Exception:
            identified(None, True)

    def finish_exchange(self, exchange, confirmed):
        if self.exchange is not exchange or not self._can_present(self.candidate_nav):
            return
        nav, old, previous = self.candidate_nav, self.view, self.observation
        if self.session.edits:
            if not exchange.asked:  # confirmed: the window keeps them; silent (busy or hung): its owner is asked, once
                if confirmed:
                    self.drop_candidate()
                exchange.asked = not confirmed
                self.held_editor(None if confirmed else exchange)
            return
        if (not nav.ticket or nav.legacy) and self.session.has_room_input:
            if confirmed and exchange.outgoing and exchange.outgoing.ticket and exchange.outgoing.legacy != "separate":
                self.refuse("the selected interface cannot restore room input", wait_for_builtin=True)
                return
            nav.legacy = "separate"
        elif not exchange.shown and confirmed:
            # The owner finished editing while the previous view still had focus.
            # Cut over and collect its final turn under the same deadline.
            if self._show_candidate(exchange):
                try:
                    self._ask_for_final_input(old, exchange)
                    if exchange.asked:  # its first deadline already passed, asking the owner
                        _on_appkit(self.finish_exchange, exchange, False, after=_DRAIN_SECONDS)
                except Exception:
                    self.finish_exchange(exchange, confirmed=False)
            return
        if not exchange.shown and not self._show_candidate(exchange):
            return
        self.view, self.navigation, self.observation = self.candidate, nav, self.candidate_observation
        self.exchange = self.candidate = self.candidate_nav = self.candidate_observation = None
        self.presented, self.pending, self.requested = nav.generation, None, False
        self.session.edits.clear()  # the previous document owned these non-room editors
        self.waiting_for_builtin = False
        self.window.real_url = self.browser.url = f"http://127.0.0.1:{nav.port}/"
        try:
            self._discard_view(old, previous)
            self.background.chrome.prepare_renderer(None)  # installed: no view is left to put back
        finally:
            try:
                if self.release_vendor_retain:  # only the first WebKitHost got pywebview 5.4's extra retain()
                    self.release_vendor_retain = False
                    old.release()
            finally:
                if nav.ready_reply is not None:
                    reply, nav.ready_reply = nav.ready_reply, None
                    reply({} if nav.legacy else self.session.items.copy(), None)
        if not confirmed:
            log.warning("WebKit previous document did not confirm its final room input")
            self.notify_previous("ouro:room-input-unconfirmed")
        log.info("WebKit main view installed in the existing window")
        self._help()

    @staticmethod
    def _discard_view(view, observation):
        try:
            if observation is not None:
                observation.dispose()
        finally:
            try:
                view.setNavigationDelegate_(None)
                view.setUIDelegate_(None)
            finally:
                try:
                    view.stopLoading()
                finally:
                    view.loadHTMLString_baseURL_("", None)  # end this document's sockets and media

    def retire_generation(self, generation):
        if self.pending is generation:
            self.pending, self.requested = None, False
        if self.candidate_nav is not None and self.candidate_nav.generation is generation:
            self.drop_candidate()
        else:
            self._help()  # no replacement is under way any more

    def refuse(self, reason, *, wait_for_builtin=False):
        generation = self.candidate_nav.generation if self.candidate_nav else self.pending
        if generation is self.pending:
            self.requested = False
            self.waiting_for_builtin = wait_for_builtin
            if not wait_for_builtin:
                self.pending = None
        self.drop_candidate()
        log.warning("WebKit replacement refused: %s; previous view and room input retained", reason)
        self.notify_previous("ouro:renderer-incompatible" if wait_for_builtin else "ouro:renderer-refused")

    def drop_candidate(self):
        view, nav, observation = self.candidate, self.candidate_nav, self.candidate_observation
        exchange = self.exchange
        self.candidate = self.candidate_nav = self.candidate_observation = self.exchange = None
        if view is not None:
            try:
                if exchange is not None and exchange.shown and self.view is not None:
                    self.browser.window.setContentView_(self.view)
                    self.browser.webview = self.view
                    self.bind_bridge(self.view, leaving=nav)
                    self._follow(self.view)
                    for event, was_set in exchange.events:
                        # Restore pywebview 5.4's wait latch without firing the old page's load hooks again.
                        (event._event.set if was_set else event.clear)()
                    if exchange.focus:
                        self.browser.window.makeFirstResponder_(self.view)
                    self.notify_previous("ouro:room-input-unconfirmed")
                try:
                    self.background.chrome.prepare_renderer(None)
                except Exception as exc:
                    log.warning("WebKit window controls could not discard the candidate (%s)", type(exc).__name__)
            finally:
                try:
                    self._discard_view(view, observation)
                finally:
                    if nav and nav.ready_reply is not None:
                        reply, nav.ready_reply = nav.ready_reply, None
                        reply(None, "This window was not selected.")
        self._help()  # the window settled on its previous view, unless a replacement is still asked for

    def _dispose(self):
        self.pending = self.view = self.navigation = self.previous_panic = self.help = None
        self.requested = self.first = False  # a closed window is owed neither its first document nor Help's page
        if self.bridge is not None:
            self.bridge.retire()
        self.drop_candidate()
        if self.observation is not None:
            self.observation.dispose()
        if self.close_observer is not None:
            from Foundation import NSNotificationCenter
            NSNotificationCenter.defaultCenter().removeObserver_(self.close_observer)
            self.close_observer = None

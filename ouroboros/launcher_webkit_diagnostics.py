"""Native evidence for a macOS desktop window. None of these observations request a reload.

Call `attach(wk_webview)` on AppKit's thread before the first navigation and retain the returned
handle (WKWebView's navigationDelegate is weak). Call its `replacement_requested()` when the
launcher decides to replace the window, and `dispose()` on AppKit's thread when retiring it.
The writer waits only for submitted events. Neither replacement nor disposal waits for the writer.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from functools import cache

log = logging.getLogger("launcher.webkit")

_PROVISIONAL = "webView:didFailProvisionalNavigation:withError:"
_COMMITTED = "webView:didFailNavigation:withError:"
_DOMAINS = {name: name for name in (
    "NSURLErrorDomain", "WKErrorDomain", "WebKitErrorDomain", "NSCocoaErrorDomain",
    "NSPOSIXErrorDomain", "NSOSStatusErrorDomain", "kCFErrorDomainCFNetwork",
)}

# WebKit UIProcess/{GPU/GPUProcessProxy,WebProcessProxy}.cpp, didBecomeUnresponsive().
# Only the unexpanded format is compared. A proxy reporting unresponsiveness does not identify a window.
_PROCESS_FORMATS = {
    "%p - GPUProcessProxy::didBecomeUnresponsive: GPUProcess with PID %d became unresponsive, terminating it": "gpu",
    "%p - [PID=%i] WebProcessProxy::didBecomeUnresponsive:": "web_content",
}
_SUBSYSTEM, _CATEGORY = "com.apple.WebKit", "Process"
_MAX_ENTRIES = 1024


def _navigation_failed(delegate, selector, stage, view, navigation, error):
    events = delegate.events
    if events is not None:
        try:
            native_domain = error.domain()
            domain = _DOMAINS.get(native_domain, "other") if native_domain is not None else "unavailable"
        except Exception:
            domain = "unavailable"
        try:
            code = error.code()  # NSError.code is an NSInteger.
            if type(code) is not int:
                code = "unavailable"
        except Exception:
            code = "unavailable"
        try:
            events.put((log.warning, ("WebKit navigation_error stage=%s domain=%s code=%s observed_at=%.3f",
                                      stage, domain, code, time.time())))
        except Exception:
            pass  # The original delegate still receives its callback.
    previous = delegate.previous
    if previous.respondsToSelector_(selector):
        getattr(previous, selector.replace(":", "_"))(view, navigation, error)


@cache
def _delegate_type():
    import objc
    from Foundation import NSObject

    class OuroborosNavigationDiagnostics(NSObject):
        def respondsToSelector_(self, selector):
            previous = getattr(self, "previous", None)
            return bool(objc.super(OuroborosNavigationDiagnostics, self).respondsToSelector_(selector)
                        or (previous is not None and previous.respondsToSelector_(selector)))

        def forwardingTargetForSelector_(self, selector):
            previous = getattr(self, "previous", None)
            if previous is not None and previous.respondsToSelector_(selector):
                return previous
            return objc.super(OuroborosNavigationDiagnostics, self).forwardingTargetForSelector_(selector)

        @objc.typedSelector(b"v@:@@@")
        def webView_didFailProvisionalNavigation_withError_(self, view, navigation, error):
            _navigation_failed(self, _PROVISIONAL, "provisional", view, navigation, error)

        @objc.typedSelector(b"v@:@@@")
        def webView_didFailNavigation_withError_(self, view, navigation, error):
            _navigation_failed(self, _COMMITTED, "committed", view, navigation, error)

    return OuroborosNavigationDiagnostics


@cache
def _oslog_types():
    # pywebview installs PyObjC Core and Cocoa, without the OSLog Python wrapper.
    # Supply the two public NSError **out arguments to the bridge.
    import objc

    objc.loadBundle("OSLog", {}, bundle_path="/System/Library/Frameworks/OSLog.framework", scan_classes=False)
    for selector, index in ((b"storeWithScope:error:", 3),
                            (b"entriesEnumeratorWithOptions:position:predicate:error:", 5)):
        objc.registerMetaDataForSelector(b"OSLogStore", selector, {"arguments": {index: {"type": b"o^@"}}})
    return objc.lookUpClass("OSLogStore"), objc.lookUpClass("OSLogEntryLog")


def _read_process_log(since, until, latest):
    import objc
    from Foundation import NSDate, NSPredicate

    with objc.autorelease_pool():
        store_class, entry_class = _oslog_types()
        store, _ = store_class.storeWithScope_error_(1, None)  # OSLogStoreCurrentProcessIdentifier
        if store is None or until < since:
            return "unavailable"
        position = store.positionWithDate_(NSDate.dateWithTimeIntervalSince1970_(since))
        predicate = NSPredicate.predicateWithFormat_argumentArray_(
            "subsystem == %@ AND category == %@", [_SUBSYSTEM, _CATEGORY])
        if position is None or predicate is None:
            return "unavailable"
        rows, _ = store.entriesEnumeratorWithOptions_position_predicate_error_(0, position, predicate, None)
        if rows is None:
            return "unavailable"
        for index, entry in enumerate(rows):
            if index == _MAX_ENTRIES:
                return "entry_limit"
            if not isinstance(entry, entry_class) or entry.subsystem() != _SUBSYSTEM or entry.category() != _CATEGORY:
                continue
            when = entry.date().timeIntervalSince1970()
            if since < when <= until:
                subject = _PROCESS_FORMATS.get(entry.formatString())
                if subject is not None:
                    latest[subject] = max(when, latest.get(subject, when))
    return "matched" if latest else "no_matching_entry"


def _record_replacement(since, until):
    log.info("WebKit window_replacement_requested requested_at=%.3f", until)
    latest = {}
    try:
        result = _read_process_log(since, until, latest)
    except Exception:
        result = "unavailable"
    for subject, when in latest.items():
        log.warning("WebKit process_unresponsive_report source=current_process_oslog subject=%s observed_at=%.3f",
                    subject, when)
    log.info("WebKit process_log_lookup scope=current_process since=%.3f until=%.3f result=%s", since, until, result)


def _write_events(events):
    for write, args in iter(events.get, None):
        try:
            write(*args)
        except Exception:
            pass  # A failing log handler must not strand later events or invoke the shell's exception hook.


class WindowDiagnostics:
    def __init__(self):
        self._since = time.time()
        self._lock = threading.Lock()
        self._installed = None
        self._events = queue.SimpleQueue()
        try:
            threading.Thread(target=_write_events, args=(self._events,),
                             name="webkit-diagnostics", daemon=True).start()
        except RuntimeError:
            self._events = None

    def replacement_requested(self) -> bool:
        """Any thread. Queue an observation since attach or the last request. False if the writer is unavailable."""
        with self._lock:
            if self._events is None:
                return False
            until = time.time()
            self._events.put((_record_replacement, (self._since, until)))
            self._since = until
            return True

    def dispose(self) -> bool:
        """AppKit thread. If restoration fails, keep the handle and retry at close. The writer can end now."""
        installed = self._installed
        if installed is not None:
            view, delegate = installed
            delegate.events = None
            try:
                if view.navigationDelegate() is delegate:
                    view.setNavigationDelegate_(delegate.previous)
                self._installed = None
            except Exception:
                if self._events is not None:
                    self._events.put((log.warning, ("WebKit navigation_observer restore=unavailable",)))
        with self._lock:
            if self._events is not None:
                self._events.put(None)
                self._events = None
        return self._installed is None


def attach(webview) -> WindowDiagnostics:
    """AppKit thread. Interpose only on this WKWebView's navigation delegate."""
    handle = WindowDiagnostics()
    if handle._events is None:
        return handle
    try:
        previous = webview.navigationDelegate()
        # pywebview 5.4 has no termination callback. Leave a custom owner of that policy in place.
        if previous is not None and not previous.respondsToSelector_("webViewWebContentProcessDidTerminate:"):
            observer = _delegate_type()
            if not isinstance(previous, observer):
                delegate = observer.alloc().init()
                delegate.previous, delegate.events = previous, handle._events
                handle._installed = webview, delegate
                webview.setNavigationDelegate_(delegate)
                return handle
    except Exception:
        pass
    handle._events.put((log.warning, ("WebKit navigation_observer unavailable",)))
    return handle

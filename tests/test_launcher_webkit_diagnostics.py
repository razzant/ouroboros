"""Source-shaped Cocoa and OSLog stand-ins. No window, process fault or system log is needed."""

import logging
import queue
import sys
import threading
from contextlib import nullcontext
from functools import cache
from types import SimpleNamespace

import pytest

from ouroboros import launcher_webkit_diagnostics as diagnostics


PROVISIONAL = "webView:didFailProvisionalNavigation:withError:"
COMMITTED = "webView:didFailNavigation:withError:"
TERMINATED = "webViewWebContentProcessDidTerminate:"
POLICY = "webView:decidePolicyForNavigationAction:decisionHandler:"
PYWEBVIEW = (
    POLICY, "webView:decidePolicyForNavigationResponse:decisionHandler:",
    "webView:navigationAction:didBecomeDownload:", "webView:didReceiveAuthenticationChallenge:completionHandler:",
    "webView:didFinishNavigation:", "webView:createWebViewWithConfiguration:forNavigationAction:windowFeatures:",
    "userContentController:didReceiveScriptMessage:",
)
GPU_FORMAT = "%p - GPUProcessProxy::didBecomeUnresponsive: GPUProcess with PID %d became unresponsive, terminating it"
WEB_FORMAT = "%p - [PID=%i] WebProcessProxy::didBecomeUnresponsive:"
PRIVATE = "https://page.invalid/private?token=test-only"


class NSObject:
    @classmethod
    def alloc(cls):
        return cls()

    def init(self):
        return self

    def respondsToSelector_(self, selector):
        return hasattr(type(self), selector.replace(":", "_"))

    def forwardingTargetForSelector_(self, selector):
        return None


class BrowserDelegate:
    def __init__(self, *extra_selectors):
        self.answers = set(PYWEBVIEW).union(extra_selectors)
        self.calls = []

    def respondsToSelector_(self, selector):
        return selector in self.answers

    def webView_decidePolicyForNavigationAction_decisionHandler_(self, view, action, handler):
        self.calls.append((view, action, handler))
        handler("original decision")

    def webView_didFailProvisionalNavigation_withError_(self, view, nav, error):
        self.calls.append((PROVISIONAL, view, nav, error))

    def webView_didFailNavigation_withError_(self, view, nav, error):
        self.calls.append((COMMITTED, view, nav, error))


class View:
    def __init__(self, delegate):
        self.current = delegate
        self.writes = []
        self.unexpected = []

    def __getattr__(self, name):
        self.unexpected.append(name)
        raise AssertionError(name)

    def navigationDelegate(self):
        return self.current

    def setNavigationDelegate_(self, delegate):
        self.writes.append(delegate)
        self.current = delegate


@pytest.fixture
def mac(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="launcher.webkit")
    state = SimpleNamespace(now=1000.0, workers=[])

    class Thread:
        def __init__(self, *, target, args, name, daemon):
            assert daemon
            self.run = lambda: target(*args)

        def start(self):
            state.workers.append(self.run)  # the caller never runs, joins or waits for the job

    class Messages(queue.SimpleQueue):
        get = queue.SimpleQueue.get_nowait  # fail immediately if disposal didn't send the end marker

    state.objc = SimpleNamespace(super=super, typedSelector=lambda signature: lambda method: method,
                                 autorelease_pool=nullcontext)
    state.foundation = SimpleNamespace(NSObject=NSObject)
    monkeypatch.setitem(sys.modules, "objc", state.objc)
    monkeypatch.setitem(sys.modules, "Foundation", state.foundation)
    for name in ("_delegate_type", "_oslog_types"):
        monkeypatch.setattr(diagnostics, name, cache(getattr(diagnostics, name).__wrapped__))
    monkeypatch.setattr(diagnostics, "threading", SimpleNamespace(Lock=threading.Lock, Thread=Thread))
    monkeypatch.setattr(diagnostics, "queue", SimpleNamespace(SimpleQueue=Messages))
    monkeypatch.setattr(diagnostics, "time", SimpleNamespace(time=lambda: state.now))
    return state


class PrivatePayload:
    """Notice even an attempted read whose exception the observer catches."""
    def __init__(self):
        self.unexpected = []

    def __getattr__(self, field):
        self.unexpected.append(field)
        raise AssertionError(field)

    def __str__(self):
        self.unexpected.append("stringify")
        return PRIVATE

    __repr__ = __str__


class Error(PrivatePayload):
    def __init__(self, domain, code):
        super().__init__()
        self.domain = lambda: domain
        self.code = lambda: code


def test_navigation_tap_preserves_the_original_selector_surface_and_completion(mac, caplog):
    previous = BrowserDelegate(PROVISIONAL)
    view = View(previous)
    handle = diagnostics.attach(view)
    observer = view.current
    assert observer is not previous
    for selector in PYWEBVIEW:
        assert observer.respondsToSelector_(selector)
        assert observer.forwardingTargetForSelector_(selector) is previous
    for absent in (TERMINATED, "webView:decidePolicyForNavigationAction:preferences:decisionHandler:"):
        assert not observer.respondsToSelector_(absent)
        assert observer.forwardingTargetForSelector_(absent) is None
    assert observer.respondsToSelector_(PROVISIONAL) and observer.respondsToSelector_(COMMITTED)
    decisions = []
    action, handler = object(), decisions.append
    observer.forwardingTargetForSelector_(POLICY).webView_decidePolicyForNavigationAction_decisionHandler_(
        view, action, handler)
    assert decisions == ["original decision"] and previous.calls == [(view, action, handler)]

    nav = PrivatePayload()
    error = Error("NSURLErrorDomain", -999)  # cancelled navigation is a native error, not a process failure
    observer.webView_didFailProvisionalNavigation_withError_(view, nav, error)
    observer.webView_didFailNavigation_withError_(view, nav, error)
    assert previous.calls[1:] == [(PROVISIONAL, view, nav, error)]
    assert handle.dispose() and handle.dispose()
    assert view.writes == [observer, previous]
    assert not view.unexpected  # UI delegate, script handlers, loading and input all remain pywebview's.
    assert not handle.replacement_requested()
    assert not caplog.records
    mac.workers[0]()
    assert caplog.messages == [
        "WebKit navigation_error stage=provisional domain=NSURLErrorDomain code=-999 observed_at=1000.000",
        "WebKit navigation_error stage=committed domain=NSURLErrorDomain code=-999 observed_at=1000.000",
    ]
    assert not nav.unexpected and not error.unexpected


def test_private_or_unreadable_error_fields_never_escape_and_the_previous_listener_still_runs(mac, caplog):
    previous = BrowserDelegate(COMMITTED)
    view = View(previous)
    handle = diagnostics.attach(view)
    private = Error(PRIVATE, PRIVATE)
    broken = Error("WKErrorDomain", 4)

    def unreadable():
        raise ValueError(PRIVATE)

    broken.code = unreadable
    for error in (private, broken):
        view.current.webView_didFailNavigation_withError_(view, None, error)
    handle.dispose()
    mac.workers[0]()
    assert previous.calls == [(COMMITTED, view, None, private), (COMMITTED, view, None, broken)]
    assert "domain=other code=unavailable" in caplog.messages[0]
    assert "domain=WKErrorDomain code=unavailable" in caplog.messages[1]
    assert PRIVATE not in caplog.text and all(not r.exc_info and not r.stack_info for r in caplog.records)
    assert not private.unexpected and not broken.unexpected


@pytest.mark.parametrize("current", [None, BrowserDelegate(TERMINATED)])
def test_an_absent_delegate_or_custom_termination_owner_is_left_alone(mac, current, caplog):
    view = View(current)
    handle = diagnostics.attach(view)
    handle.dispose()
    mac.workers[0]()
    assert view.current is current and not view.writes
    assert caplog.messages == ["WebKit navigation_observer unavailable"]


@pytest.mark.parametrize("successor", [None, BrowserDelegate()])
def test_close_does_not_resurrect_a_delegate_or_replace_its_successor(mac, successor, caplog):
    view = View(BrowserDelegate())
    handle = diagnostics.attach(view)
    ours = view.current
    duplicate = diagnostics.attach(view)
    duplicate.dispose()
    assert view.current is ours
    view.setNavigationDelegate_(successor)  # pywebview clears it on windowWillClose:
    handle.dispose()
    ours.webView_didFailNavigation_withError_(view, None, Error("WKErrorDomain", 4))
    for run in mac.workers:
        run()
    assert view.writes == [ours, successor]
    assert caplog.messages == ["WebKit navigation_observer unavailable"]


def test_a_refused_restore_can_be_retried_and_keeps_forwarding_in_the_meantime(mac, caplog):
    previous = BrowserDelegate(COMMITTED)
    view = View(previous)
    handle = diagnostics.attach(view)
    setter = view.setNavigationDelegate_

    def refuse(delegate):
        raise RuntimeError(PRIVATE)

    view.setNavigationDelegate_ = refuse
    assert not handle.dispose()
    error = Error("WKErrorDomain", 4)
    view.current.webView_didFailNavigation_withError_(view, None, error)
    assert previous.calls == [(COMMITTED, view, None, error)]
    view.setNavigationDelegate_ = setter
    assert handle.dispose() and view.current is previous
    mac.workers[0]()
    assert caplog.messages == ["WebKit navigation_observer restore=unavailable"]


def test_writer_start_failure_leaves_the_original_in_place(mac, monkeypatch, caplog):
    def refuse_start(self):
        raise RuntimeError(PRIVATE)

    monkeypatch.setattr(diagnostics.threading.Thread, "start", refuse_start)
    previous = BrowserDelegate()
    view = View(previous)
    handle = diagnostics.attach(view)
    assert not handle.replacement_requested()
    assert handle.dispose() and view.current is previous and not view.writes
    assert not mac.workers and not caplog.records


class Entry(PrivatePayload):
    def __init__(self, when, fmt, subsystem="com.apple.WebKit", category="Process"):
        super().__init__()
        self.date = lambda: SimpleNamespace(timeIntervalSince1970=lambda: when)
        self.subsystem = lambda: subsystem
        self.category = lambda: category
        self.formatString = lambda: fmt


def oslog(mac, entries, failure=None):
    calls = []
    error = PrivatePayload()  # even a native failure result must not be described

    class Store:
        @classmethod
        def storeWithScope_error_(cls, scope, out):
            calls.append(("scope", scope, out))
            return (None, error) if failure == "store" else (cls(), None)

        def positionWithDate_(self, since):
            calls.append(("position", since))
            return None if failure == "position" else since

        def entriesEnumeratorWithOptions_position_predicate_error_(self, options, position, predicate, out):
            calls.append(("entries", options, position, predicate, out))
            return (None, error) if failure == "entries" else (iter(entries), None)

    def predicate(fmt, values):
        calls.append(("predicate", fmt, tuple(values)))
        return None if failure == "predicate" else "scoped filter"

    mac.objc.loadBundle = lambda name, ns, **kwargs: calls.append(("load", name, kwargs))
    mac.objc.registerMetaDataForSelector = lambda *args: calls.append(("metadata", *args))
    mac.objc.lookUpClass = {"OSLogStore": Store, "OSLogEntryLog": Entry}.__getitem__
    mac.foundation.NSDate = SimpleNamespace(dateWithTimeIntervalSince1970_=float)
    mac.foundation.NSPredicate = SimpleNamespace(predicateWithFormat_argumentArray_=predicate)
    return calls, error


def test_process_evidence_covers_the_window_lifetime_and_uses_only_exact_unexpanded_formats(mac, caplog):
    entries = [
        object(), Entry(999, GPU_FORMAT), Entry(4700, WEB_FORMAT),
        Entry(1800, GPU_FORMAT, subsystem="com.example.browser"), Entry(2000, WEB_FORMAT, category="Loading"),
        Entry(2100, GPU_FORMAT + PRIVATE), Entry(2200, "0xab - [PID=17] WebProcessProxy::didBecomeUnresponsive:"),
        Entry(1300, GPU_FORMAT), Entry(3000, WEB_FORMAT),
    ]
    calls, _ = oslog(mac, entries)
    view = View(BrowserDelegate())
    handle = diagnostics.attach(view)
    mac.now = 4600.0
    assert handle.replacement_requested()  # the GPU report is already 55 minutes old
    mac.now = 4800.0
    assert handle.replacement_requested()
    handle.dispose()
    assert not calls and not caplog.records  # querying OSLog and writing the file both belong to the writer
    mac.workers[0]()
    assert [c for c in calls if c[0] == "scope"] == [("scope", 1, None)] * 2
    assert [c for c in calls if c[0] == "entries"] == [
        ("entries", 0, 1000.0, "scoped filter", None), ("entries", 0, 4600.0, "scoped filter", None)]
    assert ("predicate", "subsystem == %@ AND category == %@", ("com.apple.WebKit", "Process")) in calls
    assert ("load", "OSLog", {"bundle_path": "/System/Library/Frameworks/OSLog.framework",
                              "scan_classes": False}) in calls
    assert [(c[2], c[3]) for c in calls if c[0] == "metadata"] == [
        (b"storeWithScope:error:", {"arguments": {3: {"type": b"o^@"}}}),
        (b"entriesEnumeratorWithOptions:position:predicate:error:", {"arguments": {5: {"type": b"o^@"}}}),
    ]
    assert caplog.messages == [
        "WebKit window_replacement_requested requested_at=4600.000",
        "WebKit process_unresponsive_report source=current_process_oslog subject=gpu observed_at=1300.000",
        "WebKit process_unresponsive_report source=current_process_oslog subject=web_content observed_at=3000.000",
        "WebKit process_log_lookup scope=current_process since=1000.000 until=4600.000 result=matched",
        "WebKit window_replacement_requested requested_at=4800.000",
        "WebKit process_unresponsive_report source=current_process_oslog subject=web_content observed_at=4700.000",
        "WebKit process_log_lookup scope=current_process since=4600.000 until=4800.000 result=matched",
    ]
    assert all(not entry.unexpected for entry in entries if isinstance(entry, Entry))


@pytest.mark.parametrize("failure", ["framework", "store", "position", "predicate", "entries", None])
def test_request_is_recorded_independently_of_absent_evidence_and_unavailable_lookups(mac, caplog, failure):
    calls, error = oslog(mac, [], failure)
    if failure == "framework":
        def unavailable(*args, **kwargs):
            raise RuntimeError(PRIVATE)
        mac.objc.loadBundle = unavailable
    handle = diagnostics.attach(View(BrowserDelegate()))
    mac.now = 1060.0
    assert handle.replacement_requested()
    handle.dispose()
    mac.workers[0]()
    result = "unavailable" if failure else "no_matching_entry"
    assert caplog.messages == [
        "WebKit window_replacement_requested requested_at=1060.000",
        f"WebKit process_log_lookup scope=current_process since=1000.000 until=1060.000 result={result}"]
    if failure in ("store", "position", "predicate"):
        assert not any(c[0] == "entries" for c in calls)
    assert not error.unexpected and all(not r.exc_info and not r.stack_info for r in caplog.records)


@pytest.mark.parametrize("stopped", ["entry_limit", "unavailable"])
def test_bounded_or_interrupted_search_keeps_known_reports_and_marks_the_gap(mac, monkeypatch, caplog, stopped):
    def rows():
        yield Entry(1200, WEB_FORMAT)
        if stopped == "unavailable":
            raise RuntimeError(PRIVATE)
        yield Entry(1500, GPU_FORMAT)
        pytest.fail("enumerator was consumed past the bound")

    oslog(mac, rows())
    monkeypatch.setattr(diagnostics, "_MAX_ENTRIES", 1)
    handle = diagnostics.attach(View(BrowserDelegate()))
    mac.now = 2000.0
    handle.replacement_requested()
    handle.dispose()
    mac.workers[0]()
    assert len(caplog.messages) == 3
    assert "subject=web_content observed_at=1200.000" in caplog.messages[1]
    assert caplog.messages[-1].endswith("result=" + stopped)
    assert PRIVATE not in caplog.text and not any(r.exc_info for r in caplog.records)

"""Ownership on the normal AppKit dispatch path, with stand-ins for the window and both views."""
from __future__ import annotations

import json
import sys
from threading import Event
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, call

import pytest

from ouroboros import launcher_webkit
from ouroboros.desktop_room_session import PRESENTATION_HEADER, RESPONSE_HEADER, document_ticket
from ouroboros.launcher_webkit import MainRenderer


@pytest.fixture(autouse=True)
def native(monkeypatch):
    class NSObject:
        def __init_subclass__(cls, **kwargs):
            super().__init_subclass__()

        @classmethod
        def alloc(cls):
            return cls()

        def init(self):
            return self

    kit = SimpleNamespace(WKNavigationTypeBackForward=2, WKNavigationTypeReload=3,
                          WKNavigationActionPolicyCancel=0, WKContentWorld=Mock())
    host, app = Mock(), Mock()
    host.alloc.return_value.initWithFrame_configuration_.side_effect = lambda *_: Mock(name="incoming view")
    monkeypatch.setitem(sys.modules, "objc", SimpleNamespace(protocolNamed=str, python_method=lambda fn: fn, pyobjc_id=id))
    monkeypatch.setitem(sys.modules, "WebKit", kit)
    monkeypatch.setitem(sys.modules, "Foundation", SimpleNamespace(NSObject=NSObject, NSURL=Mock(), NSURLRequest=Mock()))
    cocoa = SimpleNamespace(BrowserView=SimpleNamespace(WebKitHost=host))
    platforms = SimpleNamespace(cocoa=cocoa)
    monkeypatch.setitem(sys.modules, "webview", SimpleNamespace(platforms=platforms))
    monkeypatch.setitem(sys.modules, "webview.platforms", platforms)
    monkeypatch.setitem(sys.modules, "webview.platforms.cocoa", cocoa)
    monkeypatch.setitem(sys.modules, "PyObjCTools", SimpleNamespace(AppHelper=app))
    classes = launcher_webkit._native_classes.__wrapped__()
    monkeypatch.setattr(launcher_webkit, "_native_classes", lambda: classes)
    monkeypatch.setattr(launcher_webkit.diagnostics, "attach", lambda view: Mock())
    # The registry read of a confirmed replacement is a loopback HTTP probe: a test names its answer, none asks a server.
    monkeypatch.setattr(launcher_webkit, "absent_rooms", lambda port, rooms: set())
    return SimpleNamespace(kit=kit, app=app, guard=classes[2])


def renderer(tmp_path):
    owner = MainRenderer(tmp_path, lambda proc, port: True)
    owner.view = Mock(name="initial view")
    owner.window = Mock(name="pywebview window")
    owner.window._callbacks = {}
    owner.browser = Mock(name="BrowserView")
    owner.browser.pywebview_window, owner.browser.webview = owner.window, owner.view
    owner.background, owner.observation = Mock(), Mock()
    owner.server_environment({})
    owner.confirmed(object(), 9450, replace=False)
    owner.navigation = owner._navigation(owner.view, owner.latest)
    owner.navigation.source, owner.navigation.ticket = "builtin", "first-document"
    owner.presented, owner.installed = owner.latest, True
    owner.bind_bridge(owner.view)
    owner.receive(message(owner.view, owner.navigation, "open", staged=True))
    return owner


def message(view, nav, op, *, url=None, main=True, **body):
    envelope = Mock()
    envelope.webView.return_value = view
    envelope.frameInfo.return_value.isMainFrame.return_value = main
    envelope.frameInfo.return_value.request.return_value.URL.return_value.absoluteString.return_value = (
        url or f"http://127.0.0.1:{nav.port}/")
    envelope.body.return_value = {"op": op, "ticket": nav.ticket, **body}
    return envelope


def start_candidate(owner, port):
    owner.replacement_requested()
    owner.server_environment({})
    owner.confirmed(object(), port, replace=True)
    owner._replace(owner.latest)
    return owner.candidate, owner.candidate_nav


def response_policy(view, nav, *, source="builtin", status=200, url=None, mime="text/html", signed=True, attachment=""):
    http, response, handler = Mock(), Mock(), Mock()
    handler.__block_signature__ = True
    response.response.return_value = http
    response.isForMainFrame.return_value = response.canShowMIMEType.return_value = True
    http.URL.return_value.absoluteString.return_value = url or f"http://127.0.0.1:{nav.port}/"
    http.statusCode.return_value, http.MIMEType.return_value = status, mime
    header = RESPONSE_HEADER if source == "builtin" else PRESENTATION_HEADER
    headers = {"Content-Disposition": attachment}
    if signed:
        headers[header] = document_ticket(nav.key, nav.challenge, source=source)
    http.valueForHTTPHeaderField_.side_effect = headers.get
    nav.webView_decidePolicyForNavigationResponse_decisionHandler_(view, response, handler)
    return handler


def commit_document(view, nav, source="builtin"):
    response_policy(view, nav, source=source, status=404 if source == "none" else 200)
    nav.webView_didCommitNavigation_(view, nav.expected_navigation)


def candidate(owner, port):
    view, nav = start_candidate(owner, port)
    commit_document(view, nav)
    return view, nav


def ready(owner, view, nav, project="", **facts):
    receiver = launcher_webkit._native_classes()[1]()
    receiver.owner = owner
    reply = Mock()
    event = message(view, nav, "ready", project=project, **facts)
    receiver.userContentController_didReceiveScriptMessage_replyHandler_(None, event, reply)
    return reply


def drained(owner, items, editors=()):
    return owner.receive(message(owner.view, owner.navigation, "drained", handoff=owner.exchange.token,
                                 items=json.dumps(items), editors=editors))


def queued(native, turn):
    """The arguments of each AppKit turn queued for ``turn``, in order (the lifecycle owner's words queue their own)."""
    return [queue.args[1:] for queue in native.app.callAfter.call_args_list if queue.args[0] == turn]


def test_two_restarts_keep_the_window_and_retire_only_the_issuing_view(tmp_path):
    owner = renderer(tmp_path)
    first = owner.view
    window, browser, native = owner.window, owner.browser, owner.browser.window
    owner.session.update("ouro_chat_draft:7", "  exact\n")
    accepted = []
    for port in (9451, 9452):
        old, old_bridge, observer = owner.view, owner.bridge, owner.observation
        window._callbacks["waiting"] = Mock()
        waiter = Event()
        old_bridge.waiting.add(waiter)
        new, nav = candidate(owner, port)
        accepted.append(new)
        native.firstResponder.return_value = old if old is first else None
        assert owner.receive(message(new, nav, "open", staged=True)) == {"mountProject": ""}
        assert browser.webview is old
        reply = ready(owner, new, nav)
        assert browser.webview is old  # WebKit can deliver the page message before didFinish.
        nav.webView_didFinishNavigation_(new, nav.expected_navigation)
        reply.assert_not_called()
        assert drained(owner, {"ouro_chat_draft:7": "  exact\n"})
        reply.assert_called_once_with({"ouro_chat_draft:7": "  exact\n"}, None)
        assert owner.window is window and owner.browser is browser and browser.webview is new
        assert window.real_url == browser.url == f"http://127.0.0.1:{port}/"
        assert owner.pending is None and not owner.requested
        assert not old_bridge.active and waiter.is_set() and window._callbacks == {}
        observer.replacement_requested.assert_called_once()
        observer.dispose.assert_called_once()
        assert "ouro:room-handoff" in old.evaluateJavaScript_completionHandler_.call_args.args[0]
        old.stopLoading.assert_called_once()
        nav.previous.webView_didFinishNavigation_.assert_called_with(new, nav.expected_navigation)
    assert native.setContentView_.call_args_list == [call(view) for view in accepted]
    assert owner.background.zoom.renderer_changed.call_args_list == [call(view) for view in accepted]
    assert owner.background.chrome.renderer_changed.call_args_list == [call(view) for view in accepted]
    native.makeFirstResponder_.assert_called_once_with(accepted[0])
    native.orderFront_.assert_not_called()
    native.close.assert_not_called()
    window.destroy.assert_not_called()
    first.release.assert_called_once()
    accepted[0].release.assert_not_called()


def test_stale_generation_or_unsaved_editor_leaves_old_input_in_place(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "AppKit", SimpleNamespace(NSAlert=Mock()))  # the held edit's native sheet
    owner = renderer(tmp_path)
    view, nav = candidate(owner, 9451)
    owner.session.update("ouro_chat_draft", "mine")
    owner.current = lambda proc, port: False
    with pytest.raises(ValueError, match="superseded"):
        owner.receive(message(view, nav, "open", staged=True))
    assert owner.candidate is None and not owner.requested
    view.stopLoading.assert_called_once()
    owner.current = lambda proc, port: True
    view, nav = candidate(owner, 9452)
    owner.session.edits.add("Files")
    owner.receive(message(view, nav, "open", staged=True))
    nav.webView_didFinishNavigation_(view, nav.expected_navigation)
    reply = ready(owner, view, nav)
    assert owner.browser.webview is owner.view
    drained(owner, {"ouro_chat_draft": "mine"}, ["Files"])
    assert reply.call_count == 1 and reply.call_args.args[0] is None
    assert owner.session.items == {"ouro_chat_draft": "mine"}
    assert owner.view is not view and owner.candidate is None
    assert owner.pending is nav.generation and not owner.requested
    owner.browser.window.setContentView_.assert_not_called()


def test_a_held_edit_is_explained_by_the_shell_even_if_its_page_takes_no_script(native, tmp_path, monkeypatch):
    owner = renderer(tmp_path)
    old = owner.view
    alert = Mock()
    monkeypatch.setitem(sys.modules, "AppKit", SimpleNamespace(NSAlert=alert))
    owner.session.edits.update({"Your rules", "Files"})
    new, nav = candidate(owner, 9451)
    owner.receive(message(new, nav, "open", staged=True))
    nav.webView_didFinishNavigation_(new, nav.expected_navigation)
    ready(owner, new, nav)
    assert owner.browser.webview is old  # the edited window stays on screen
    timers = native.app.callLater.call_count
    old.evaluateJavaScript_completionHandler_.side_effect = RuntimeError("this page takes no script")
    drained(owner, {}, ["Files", "Your rules"])
    sheet = alert.alloc().init()
    sheet.beginSheetModalForWindow_completionHandler_.assert_called_once_with(owner.browser.window, None)
    assert sheet.setMessageText_.call_args.args[0] == "Your unsaved edits are still in this window"
    words = sheet.setInformativeText_.call_args.args[0]
    assert "Files, Your rules" in words and "Nothing was saved or discarded" in words
    # The edited document stays as it is: nothing is discarded, saved, reloaded or put on a timer.
    assert owner.view is old and owner.candidate is None and owner.session.edits == {"Files", "Your rules"}
    assert owner.pending is nav.generation and not owner.requested
    old.loadHTMLString_baseURL_.assert_not_called()
    old.stopLoading.assert_not_called()
    assert native.app.callLater.call_count == timers
    assert queued(native, owner._replace) == []
    for name in ("Files", "Your rules"):  # only the owner finishing these edits asks for the fresh window again
        owner.receive(message(old, owner.navigation, "editor", name=name, dirty=False))
    assert queued(native, owner._replace) == [(nav.generation,)]


def test_zoom_and_title_bar_follow_the_presented_view_before_before_load_and_return_when_it_is_reversed(tmp_path):
    owner = renderer(tmp_path)
    old, order = owner.view, []
    owner.background.zoom.renderer_changed.side_effect = lambda view: order.append(("zoom", view))
    owner.background.chrome.renderer_changed.side_effect = lambda view: order.append(("title bar", view))
    new, nav = candidate(owner, 9451)
    # pywebview 5.4 fires before_load first in its own didFinish; there the title bar asks whether the document declared.
    nav.previous.webView_didFinishNavigation_.side_effect = lambda view, _navigation: order.append(("before_load", view))
    owner.receive(message(new, nav, "open", staged=True))
    nav.webView_didFinishNavigation_(new, nav.expected_navigation)
    reply = ready(owner, new, nav)
    assert owner.browser.webview is new and owner.view is old  # presented; its exchange is still open
    assert order == [("zoom", new), ("title bar", new), ("before_load", new)]
    owner.refuse("inert fixture reverses the presented successor")
    assert order[3:] == [("zoom", old), ("title bar", old)]
    assert owner.browser.window.setContentView_.call_args_list == [call(new), call(old)]
    assert owner.browser.webview is old and owner.view is old and owner.candidate is None
    assert owner.background.chrome.prepare_renderer.call_args_list == [call(new), call(None)]
    reply.assert_called_once_with(None, "This window was not selected.")


def test_only_the_current_main_document_can_open_or_update_its_slots(tmp_path):
    owner = renderer(tmp_path)
    nav = owner.navigation
    good = message(owner.view, nav, "update", changes=json.dumps([["ouro_chat_draft", "room"]]))
    assert owner.receive(good) is True
    assert owner.receive(message(owner.view, nav, "resume_queue")) is True
    owner.replacement_requested()
    assert owner.receive(message(owner.view, nav, "resume_queue")) is False
    owner.confirmed(object(), 9450, replace=True)
    owner.refuse("incomplete successor")
    assert not owner.requested and not owner.receive(message(owner.view, nav, "resume_queue"))
    for kwargs in ({"main": False}, {"url": "http://127.0.0.1:9450/onboarding"},
                   {"url": "http://127.0.0.1:9452/"}, {"ticket": "different-document"}):
        with pytest.raises(ValueError, match="unverified"):
            owner.receive(message(owner.view, nav, "open", **kwargs))
    with pytest.raises(ValueError, match="unverified"):
        owner.receive(message(Mock(), nav, "open"))
    nav.ticket = ""
    with pytest.raises(ValueError, match="unverified"):
        owner.receive(message(owner.view, nav, "open"))


def test_only_an_ordinary_reconnect_to_the_shown_endpoint_reopens_the_queue(tmp_path):
    owner = renderer(tmp_path)
    nav = owner.navigation
    generation = nav.generation

    def may_flush():
        return owner.receive(message(owner.view, nav, "resume_queue"))

    owner.server_exited(generation[0])
    owner.server_environment({})
    owner.confirmed(object(), nav.port, replace=False)
    owner.current = lambda proc, port: (proc, port) == owner.latest[:2]
    assert nav.generation is generation and generation is not owner.latest
    assert may_flush()
    owner.confirmed(object(), nav.port + 1, replace=False)
    assert not may_flush()
    owner.confirmed(object(), nav.port, replace=True)
    owner.refuse("deliberate replacement failed")
    assert not may_flush()
    owner.confirmed(object(), nav.port, replace=False)
    assert not may_flush()  # a later crash cannot turn a refused explicit replacement into a continuation


def action(current, target, kind=0, tag=None):
    frame, request, act = Mock(), Mock(), Mock()
    frame.request.return_value.URL.return_value.absoluteString.return_value = current
    frame.isMainFrame.return_value = True
    act.targetFrame.return_value, act.request.return_value = frame, request
    request.HTTPMethod.return_value = "GET"
    request.URL.return_value.absoluteString.return_value = target
    request.valueForHTTPHeaderField_.return_value = tag  # the shell's navigation challenge, when the request carries one
    act.shouldPerformDownload.return_value, act.navigationType.return_value = False, kind
    return act


def test_root_refresh_anchor_download_and_history_traversal(native, tmp_path):
    owner = renderer(tmp_path)
    nav, kit = owner.navigation, native.kit
    root = "http://127.0.0.1:9450/"
    refresh = root + "?_ouro_refresh=123&_ouro_reason=sha-change"
    nav.port, nav.key, nav.challenge = 9450, "a" * 64, ""
    nav.previous = Mock()
    view = owner.view
    handler = Mock()
    handler.__block_signature__ = True
    original = nav.previous.webView_decidePolicyForNavigationAction_decisionHandler_
    decide = lambda act: nav.webView_decidePolicyForNavigationAction_decisionHandler_(view, act, handler)

    owner.replacement_requested()
    decide(action(root, refresh + "#chat"))
    handler.assert_called_once_with(kit.WKNavigationActionPolicyCancel)
    view.loadRequest_.assert_not_called()
    owner.server_exited(object())  # the successor exited before publishing a port
    decide(action(root, refresh + "#chat"))
    view.loadRequest_.assert_called_once()
    for target in (refresh + "#files", root + "onboarding", "http://127.0.0.1:9451/"):
        act = action(refresh, target)
        decide(act)
        original.assert_called_with(view, act, handler)
    download = action(refresh, root)
    download.shouldPerformDownload.return_value = True
    decide(download)
    original.assert_called_with(view, download, handler)

    back = action(root + "onboarding?builtin=1", refresh, kit.WKNavigationTypeBackForward)
    decide(back)
    original.assert_called_with(view, back, handler)
    assert view.loadRequest_.call_count == 1  # Back takes its actual history step
    view.URL.return_value = back.request().URL()
    nav.webView_didCommitNavigation_(view, object())
    assert view.loadRequest_.call_count == 2
    request = back.request().mutableCopy()
    request.setValue_forHTTPHeaderField_.assert_called_once_with(nav.challenge, launcher_webkit.REQUEST_HEADER)
    response_policy(view, nav, url=refresh)
    nav.webView_didCommitNavigation_(view, nav.expected_navigation)
    owner.navigation = nav
    owner.session.update("ouro_chat_reply", "room reply")
    assert owner.receive(message(view, nav, "open", url=refresh)) == {"ouro_chat_reply": "room reply"}


def test_socket_refresh_after_failed_replacement_retries_offscreen_at_confirmed_port(native, tmp_path):
    owner = renderer(tmp_path)
    old, prior = owner.view, owner.navigation
    new, nav = start_candidate(owner, 9451)
    response_policy(new, nav, status=500)
    handler = Mock()
    handler.__block_signature__ = True
    root = "http://127.0.0.1:9450/"
    prior.webView_decidePolicyForNavigationAction_decisionHandler_(
        old, action(root, root + "?_ouro_refresh=123"), handler)
    handler.assert_called_once_with(native.kit.WKNavigationActionPolicyCancel)
    old.loadRequest_.assert_not_called()
    resume, generation = native.app.callAfter.call_args.args
    resume(generation)
    assert owner.candidate_nav.port == 9451 and owner.browser.webview is old


@pytest.fixture
def startup_clock(native, monkeypatch):
    now, callbacks = 0, []
    monkeypatch.setattr(launcher_webkit, "monotonic", lambda: now)
    native.app.callLater.side_effect = lambda seconds, fn, *args: callbacks.append((now + seconds, fn, args))

    def advance(to):
        nonlocal now
        assert now <= to
        while callbacks:
            callbacks.sort(key=lambda item: item[0])
            if callbacks[0][0] > to:
                break
            now, fn, args = callbacks.pop(0)
            fn(*args)
        now = to

    return advance


@pytest.mark.parametrize("arrived", ["room_ready", "navigation_finished", "import_then_silent"])
def test_startup_deadline_keeps_old_view_if_successor_goes_silent(tmp_path, caplog, startup_clock, arrived):
    owner = renderer(tmp_path)
    old = owner.view
    if arrived == "import_then_silent":
        owner.session.update("ouro_chat_active_project", "p7")
    view, nav = candidate(owner, 9451)
    startup_clock(17)
    owner.receive(message(view, nav, "open", staged=True))
    if arrived == "room_ready":
        reply = ready(owner, view, nav)
        reply.assert_not_called()
    else:
        nav.webView_didFinishNavigation_(view, nav.expected_navigation)
    if arrived == "import_then_silent":
        startup_clock(31)
        # The old, unqualified progress spelling is still recognized by the current launcher.
        assert owner.receive(message(view, nav, "project_importing", project="p7"))
    deadline = (31 if arrived == "import_then_silent" else 17) + launcher_webkit._STARTUP_SECONDS
    startup_clock(deadline - 1)
    assert owner.candidate is view and owner.browser.webview is old
    startup_clock(deadline)
    assert owner.view is old and owner.pending is None and not owner.requested
    if arrived == "room_ready":
        assert reply.call_count == 1 and reply.call_args.args[0] is None
    assert "timed out" in caplog.text
    assert "ouro:renderer-refused" in old.evaluateJavaScript_completionHandler_.call_args.args[0]
    view.loadHTMLString_baseURL_.assert_called_once_with("", None)
    new, other = candidate(owner, 9452)
    owner.startup_expired(nav)  # old timer and old WebKit events cannot refuse or install the later candidate
    nav.webView_didFinishNavigation_(view, nav.expected_navigation)
    assert owner.candidate is new and owner.pending is other.generation and owner.requested
    owner.browser.window.setContentView_.assert_not_called()


def test_slow_staged_room_is_presented_across_app_and_import_deadlines(tmp_path, startup_clock):
    owner = renderer(tmp_path)
    old = owner.view
    slots = {"ouro_chat_active_project": "p7", "ouro_chat_draft:7": "kept in Seven"}
    owner.session.apply(list(slots.items()))
    view, nav = start_candidate(owner, 9451)
    startup_clock(18)
    commit_document(view, nav)
    assert owner.receive(message(view, nav, "open", staged=True)) == {"mountProject": "p7"}
    startup_clock(36)
    nav.webView_didFinishNavigation_(view, nav.expected_navigation)
    for at, stage in ((37, "mount"), (48, "migrating"), (58, "boot"), (70, "migrating"), (80, "boot"),
                      (92, "migrating"), (102, "boot")):
        startup_clock(at)
        assert owner.candidate is view and owner.browser.webview is old
        assert owner.receive(message(view, nav, "project_importing", project="p7", stage=stage)) is True
    startup_clock(114)
    reply = ready(owner, view, nav, project="p7")
    assert owner.browser.webview is view
    reply.assert_not_called()
    assert drained(owner, slots)
    reply.assert_called_once_with(slots, None)
    startup_clock(140)  # no timer owns the accepted document
    assert owner.view is view and owner.pending is None and not owner.requested


def test_room_progress_cannot_keep_an_outdated_project_candidate(native, tmp_path, startup_clock):
    owner = renderer(tmp_path)
    old, outgoing = owner.view, owner.navigation
    owner.session.update("ouro_chat_active_project", "p7")
    view, nav = candidate(owner, 9451)
    assert owner.receive(message(view, nav, "open", staged=True)) == {"mountProject": "p7"}
    startup_clock(11)
    assert owner.receive(message(view, nav, "project_importing", project="p7", stage="mount"))
    with pytest.raises(ValueError, match="obsolete project read"):
        owner.receive(message(view, nav, "project_importing", project="p7", stage="heartbeat"))
    owner.receive(message(old, outgoing, "update", changes='[["ouro_chat_active_project","p8"]]'))
    with pytest.raises(ValueError, match="obsolete project read"):
        owner.receive(message(view, nav, "project_importing", project="p7", stage="migrating"))
    assert owner.candidate is None and owner.view is old and owner.pending is nav.generation
    replace, generation = native.app.callAfter.call_args.args
    replace(generation)
    replacement, current = owner.candidate, owner.candidate_nav
    commit_document(replacement, current)
    assert owner.receive(message(replacement, current, "open", staged=True)) == {"mountProject": "p8"}
    startup_clock(20)  # this was the old candidate's cutoff
    assert owner.browser.webview is old and owner.candidate is replacement


def test_selected_project_receives_the_outgoing_documents_last_send_reply_and_draft(tmp_path):
    owner = renderer(tmp_path)
    old, previous = owner.view, owner.navigation
    owner.session.apply([("ouro_chat_active_project", "p7"), ("ouro_chat_draft:7", "before Send"),
                         ("ouro_chat_files:7", "1"), ("ouro_chat_draft", "Main")])
    new, nav = candidate(owner, 9451)
    assert owner.receive(message(new, nav, "open", staged=True)) == {"mountProject": "p7"}
    nav.webView_didFinishNavigation_(new, nav.expected_navigation)
    with pytest.raises(ValueError):
        owner.receive(message(new, nav, "project_importing", project="another"))
    wrong = ready(owner, new, nav, project="another")
    assert wrong.call_args.args[0] is None and owner.browser.webview is old
    with pytest.raises(ValueError):
        owner.receive(message(new, nav, "update", changes='[["ouro_chat_draft","overwritten"]]'))
    reply = ready(owner, new, nav, project="p7")
    assert owner.browser.webview is new
    reply.assert_not_called()
    with pytest.raises(ValueError, match="outgoing"):
        owner.receive(message(old, previous, "open"))

    held = json.dumps([{"frame": {"type": "chat", "content": "before Send", "client_message_id": "original",
                                 "sender_session_id": "old", "project_id": "p7", "chat_id": 7,
                                 "reply_to": "row:7@old", "attachments": [{"filename": "sent.png"}]}}])
    # Both a late incremental update and the authoritative final snapshot arrive after the new input mounted.
    owner.receive(message(old, previous, "update", changes=json.dumps([["ouro_chat_unconfirmed:7", held]])))
    final = {"ouro_chat_active_project": "p7", "ouro_chat_draft": "Main", "ouro_chat_draft:7": "  next\n",
             "ouro_chat_reply:7": '{"address":"row:7@next"}', "ouro_chat_unconfirmed:7": held}
    assert drained(owner, final)
    reply.assert_called_once_with(final, None)
    assert owner.session.items == final  # the previous staged file count also disappeared
    with pytest.raises(ValueError):
        owner.receive(message(old, previous, "update", changes='[["ouro_chat_draft:7","older"]]'))


def test_a_second_document_cannot_inherit_the_candidates_ready_reply(tmp_path):
    owner = renderer(tmp_path)
    old = owner.view
    owner.session.update("ouro_chat_draft", "private")
    new, nav = candidate(owner, 9451)
    owner.receive(message(new, nav, "open", staged=True))
    reply = ready(owner, new, nav)
    nav.expected_navigation = object()
    nav.webView_didCommitNavigation_(new, nav.expected_navigation)
    nav.webView_didFinishNavigation_(new, nav.expected_navigation)
    assert owner.view is old and owner.candidate is None and not owner.requested
    reply.assert_called_once_with(None, "This window was not selected.")
    owner.browser.window.setContentView_.assert_not_called()


def test_selection_shell_presents_without_moving_the_unresolved_projects_input(tmp_path, startup_clock):
    owner = renderer(tmp_path)
    slots = {"ouro_chat_active_project": "p7", "ouro_chat_draft:7": "private words", "ouro_chat_reply:7": "quote"}
    owner.session.apply(list(slots.items()))
    view, nav = candidate(owner, 9451)
    assert owner.receive(message(view, nav, "open", staged=True)) == {"mountProject": "p7"}
    nav.webView_didFinishNavigation_(view, nav.expected_navigation)
    assert ready(owner, view, nav, selection=True).call_args.args[0] is None  # selection is not a Main address
    reply = ready(owner, view, nav, project="p7", selection=True)
    assert drained(owner, slots)
    startup_clock(100)  # no further Project or WebKit progress is needed to keep the installed shell
    reply.assert_called_once_with(slots, None)
    owner.browser.window.setContentView_.assert_called_once_with(view)
    assert owner.view is view and owner.session.items == slots and "ouro_chat_draft" not in owner.session.items


def test_wedged_outgoing_page_returns_last_received_input_and_warns_in_new_view(native, tmp_path, caplog):
    owner = renderer(tmp_path)
    old = owner.view
    owner.session.update("ouro_chat_draft", "last delivered")
    new, nav = candidate(owner, 9451)
    owner.receive(message(new, nav, "open", staged=True))
    nav.webView_didFinishNavigation_(new, nav.expected_navigation)
    reply = ready(owner, new, nav)
    assert native.app.callLater.call_count == 2  # one startup deadline, one final collection
    seconds, expire, *args = native.app.callLater.call_args.args
    assert 0 < seconds < native.app.callLater.call_args_list[0].args[0]
    reply.assert_not_called()
    expire(*args)
    reply.assert_called_once_with({"ouro_chat_draft": "last delivered"}, None)
    assert owner.view is new and not owner.requested and owner.pending is None
    assert "did not confirm" in caplog.text and "last delivered" not in caplog.text
    assert "ouro:room-input-unconfirmed" in new.evaluateJavaScript_completionHandler_.call_args.args[0]
    old.loadHTMLString_baseURL_.assert_called_once_with("", None)
    expire(*args)
    assert reply.call_count == 1 and native.app.callLater.call_count == 2


@pytest.mark.parametrize("response", [
    {"status": 404}, {"status": 500}, {"url": "http://127.0.0.1:9451/onboarding"},
    {"url": "http://127.0.0.1:9452/"}, {"mime": "text/plain"}, {"attachment": "attachment; filename=x"},
    {"source": "directory", "status": 404}, {"source": "none", "status": 200},
])
def test_wrong_response_keeps_even_an_empty_usable_view(tmp_path, native, response):
    owner = renderer(tmp_path)
    old = owner.view
    new, nav = start_candidate(owner, 9451)
    policy = response_policy(new, nav, **response)
    policy.assert_called_once_with(native.kit.WKNavigationActionPolicyCancel)
    assert owner.view is old and owner.browser.webview is old and owner.candidate is None
    assert not owner.requested and not owner.waiting_for_builtin
    owner.browser.window.setContentView_.assert_not_called()


@pytest.mark.parametrize("status,reported", [(200, None), (200, "none"), (404, "builtin"), (404, "directory")])
def test_older_unmarked_error_document_cannot_take_over(tmp_path, status, reported):
    owner = renderer(tmp_path)
    old = owner.view
    view, nav = start_candidate(owner, 9451)
    response_policy(view, nav, status=status, signed=False)
    nav.webView_didCommitNavigation_(view, nav.expected_navigation)
    nav.webView_didFinishNavigation_(view, nav.expected_navigation)
    script, args, _, _, checked = view.callAsyncJavaScript_arguments_inFrame_inContentWorld_completionHandler_.call_args.args
    assert args == {"stateURL": "http://127.0.0.1:9451/api/state"} and "served_ui" in script
    checked(reported, None)
    assert owner.browser.webview is old and owner.candidate is None and not owner.requested


@pytest.mark.parametrize("source,status", [("builtin", 200), ("directory", 200), ("none", 404)])
def test_older_selected_interface_can_open_after_disposition_without_a_room_ticket(tmp_path, native, source, status):
    owner = renderer(tmp_path)
    prior, old = owner.navigation, owner.view
    owner.session.update("ouro_chat_draft", "private")

    def boot(view, nav):
        response_policy(view, nav, status=status, signed=False)
        nav.webView_didCommitNavigation_(view, nav.expected_navigation)
        nav.webView_didFinishNavigation_(view, nav.expected_navigation)
        check = view.callAsyncJavaScript_arguments_inFrame_inContentWorld_completionHandler_
        check.call_args.args[-1](source, None)
        if source == "builtin":
            assert owner.browser.webview is old
            check.call_args.args[-1](True, None)  # the older app module initialized its own input
        with pytest.raises(ValueError, match="unverified"):
            owner.receive(message(view, nav, "open"))

    view, nav = start_candidate(owner, 9451)
    boot(view, nav)
    assert drained(owner, owner.session.items.copy())
    assert owner.browser.webview is old and owner.waiting_for_builtin
    owner.receive(message(old, prior, "update", changes='[["ouro_chat_draft",null]]'))
    resume, generation = native.app.callAfter.call_args.args
    resume(generation)
    view, nav = owner.candidate, owner.candidate_nav
    boot(view, nav)
    assert drained(owner, {})
    assert owner.view is view and not owner.navigation.ticket


@pytest.mark.parametrize("source", ["directory", "none"])
def test_selected_external_presentation_switches_after_owner_clears_their_room_input(native, tmp_path, source):
    owner = renderer(tmp_path)
    old, prior = owner.view, owner.navigation
    owner.session.update("ouro_chat_reply", "private quotation")
    new, nav = start_candidate(owner, 9451)
    commit_document(new, nav, source)
    with pytest.raises(ValueError):
        owner.receive(message(new, nav, "open", staged=True))
    nav.webView_didFinishNavigation_(new, nav.expected_navigation)
    assert owner.browser.webview is old
    assert drained(owner, {"ouro_chat_reply": "private quotation"})
    assert owner.view is old and owner.waiting_for_builtin
    assert "ouro:renderer-incompatible" in old.evaluateJavaScript_completionHandler_.call_args.args[0]
    owner.browser.window.setContentView_.assert_not_called()
    owner.receive(message(old, prior, "update", changes='[["ouro_chat_reply",null]]'))
    assert queued(native, owner._replace) == [(nav.generation,)]
    resume, generation = native.app.callAfter.call_args.args
    assert generation is nav.generation
    resume(generation)
    restored, presentation = owner.candidate, owner.candidate_nav
    commit_document(restored, presentation, source)
    assert owner.browser.webview is old  # even this known presentation must finish the right navigation
    presentation.webView_didFinishNavigation_(restored, presentation.expected_navigation)
    assert drained(owner, {})
    assert owner.view is restored and owner.window.real_url == "http://127.0.0.1:9451/"
    with pytest.raises(ValueError):
        owner.receive(message(restored, presentation, "open", staged=True))


@pytest.mark.parametrize("legacy,released", [(False, False), (True, False), (False, True)],
                         ids=["document-argument", "older-tagged-request", "released-webview"])
def test_only_an_already_posted_main_frame_panic_can_cross_retirement(native, tmp_path, legacy, released):
    owner = renderer(tmp_path)  # its background is a Mock: no live stop, server or process owner
    old, prior, old_bridge = owner.view, owner.navigation, owner.bridge
    challenge = "b" * 64
    prior.ticket = document_ticket(prior.key, challenge)
    guard, vendor = native.guard(), owner.browser.js_bridge
    guard.owner, guard.is_bridge, guard.original = owner, True, vendor

    def gesture(view=old, method="panic_stop", ticket=prior.ticket, **frame):
        event = message(view, prior, "", **frame)
        event.frameInfo().request().valueForHTTPHeaderField_.return_value = challenge if ticket == prior.ticket else ""
        event.body.return_value = json.dumps({"funcName": method, "params": [] if legacy else [ticket], "id": "press"})
        return event

    pressed = gesture()  # issued before the AppKit turn installs the new input
    new, nav = candidate(owner, 9451)
    guard.userContentController_didReceiveScriptMessage_(None, gesture(new))
    owner.receive(message(new, nav, "open", staged=True))
    nav.webView_didFinishNavigation_(new, nav.expected_navigation)
    reply = ready(owner, new, nav)
    reply.assert_not_called()  # the stop is independent of the final data reply
    if released:
        assert drained(owner, {})
        pressed.webView.return_value = None  # WebKit's weak reference is gone, its tagged frame request survives
    anonymous = gesture(None)
    anonymous.frameInfo().request().valueForHTTPHeaderField_.return_value = None
    for rejected in (gesture(method="request_attention"), gesture(main=False), gesture(ticket="unrelated"),
                     gesture(url="http://127.0.0.1:9450/onboarding"), gesture(Mock(name="foreign view")), anonymous):
        guard.userContentController_didReceiveScriptMessage_(None, rejected)
    owner.background.request_panic.assert_not_called()
    guard.userContentController_didReceiveScriptMessage_(None, pressed)
    owner.background.request_panic.assert_called_once_with("desktop_bridge")
    # Straight to the launcher's stop: the retired document's promise is gone, so the vendor bridge is never asked.
    vendor.userContentController_didReceiveScriptMessage_.assert_not_called()
    assert not old_bridge.active and vendor.window is owner.bridge and owner.bridge.active
    if not released:
        assert drained(owner, {})
    guard.userContentController_didReceiveScriptMessage_(None, pressed)
    assert owner.background.request_panic.call_count == 1


def room_panic(owner, view, nav, *, main=True, url=None, ticket=None):
    receiver = launcher_webkit._native_classes()[1]()
    receiver.owner = owner
    incoming = message(view, nav, "", main=main, url=url)
    incoming.name.return_value = launcher_webkit.PANIC_HANDLER
    incoming.body.return_value = nav.ticket if ticket is None else ticket
    reply = Mock()
    receiver.userContentController_didReceiveScriptMessage_replyHandler_(None, incoming, reply)
    return reply


def test_room_panic_needs_the_presented_main_document_but_not_its_ordinary_bridge(tmp_path):
    owner = renderer(tmp_path)
    old = owner.view
    view, nav = candidate(owner, 9451)
    owner.receive(message(view, nav, "open", staged=True))
    for scope in ({}, {"main": False}, {"url": "http://127.0.0.1:9451/onboarding"}, {"ticket": "other"}):
        assert room_panic(owner, view, nav, **scope).call_args.args[0] is None
    owner.background.request_panic.assert_not_called()
    nav.webView_didFinishNavigation_(view, nav.expected_navigation)
    handoff = ready(owner, view, nav)
    assert owner.view is old and owner.browser.webview is view  # ordinary room data is still with the old document
    handoff.assert_not_called()
    for scope in ({"main": False}, {"url": "http://127.0.0.1:9451/onboarding"}, {"ticket": "other"}):
        assert room_panic(owner, view, nav, **scope).call_args.args[0] is None
    owner.background.request_panic.assert_not_called()
    room_panic(owner, view, nav).assert_called_once_with({"status": "stopping"}, None)
    owner.background.request_panic.assert_called_once_with("desktop_bridge")
    owner.browser.js_bridge.userContentController_didReceiveScriptMessage_.assert_not_called()
    handoff.assert_not_called()


def test_room_panic_is_still_available_after_a_successor_fails_before_presentation(tmp_path):
    owner = renderer(tmp_path)
    old, outgoing = owner.view, owner.navigation
    view, nav = candidate(owner, 9451)
    owner.refuse("inert fixture refuses this successor")
    assert owner.view is old and owner.browser.webview is old
    assert room_panic(owner, view, nav).call_args.args[0] is None
    owner.background.request_panic.assert_not_called()
    room_panic(owner, old, outgoing).assert_called_once_with({"status": "stopping"}, None)
    owner.background.request_panic.assert_called_once_with("desktop_bridge")


@pytest.mark.parametrize("transport", ["reply", "bridge"])
@pytest.mark.parametrize("verified_next", [False, True])
def test_posted_panic_survives_a_same_view_commit_without_retaining_room_access(native, tmp_path, transport, verified_next):
    owner = renderer(tmp_path)
    view, nav = owner.view, owner.navigation
    nav.load_tagged(view, Mock())
    challenge = nav.challenge
    commit_document(view, nav)
    ticket, old_bridge = nav.ticket, owner.bridge
    owner.session.update("ouro_chat_draft:7", "kept")
    receiver, guard = launcher_webkit._native_classes()[1](), native.guard()
    vendor = owner.browser.js_bridge
    receiver.owner = guard.owner = owner
    guard.is_bridge, guard.original = True, vendor

    def posted(mark, *, main=True):
        incoming = message(view, nav, "", main=main)
        incoming.frameInfo().request().valueForHTTPHeaderField_.return_value = mark
        incoming.name.return_value = launcher_webkit.PANIC_HANDLER
        incoming.body.return_value = (ticket if transport == "reply" else
                                      json.dumps({"funcName": "panic_stop", "params": [ticket], "id": "press"}))
        return incoming

    def deliver(incoming):
        if transport == "bridge":
            guard.userContentController_didReceiveScriptMessage_(None, incoming)
        else:
            reply = Mock()
            receiver.userContentController_didReceiveScriptMessage_replyHandler_(None, incoming, reply)
            return reply.call_args.args[0]

    gesture = posted(challenge)
    # A restart at the same address may rotate the server key before the next commit.
    owner.server_environment({})
    owner.confirmed(object(), nav.port, replace=False)
    nav.load_tagged(view, Mock())
    next_challenge = nav.challenge
    assert next_challenge != challenge
    response_policy(view, nav, signed=verified_next)
    nav.webView_didCommitNavigation_(view, nav.expected_navigation)
    assert bool(nav.ticket) is verified_next and nav.ticket != ticket
    with pytest.raises(ValueError, match="unverified document"):
        owner.receive(message(view, nav, "open", ticket=ticket))
    for unrelated in (posted(next_challenge), posted(None), posted(challenge, main=False)):
        assert deliver(unrelated) is None
    owner.background.request_panic.assert_not_called()
    result = deliver(gesture)
    if transport == "reply":
        assert result == {"status": "stopping"}
    deliver(gesture)
    owner.background.request_panic.assert_called_once_with("desktop_bridge")
    vendor.userContentController_didReceiveScriptMessage_.assert_not_called()  # either door: straight to the launcher
    assert not old_bridge.active and vendor.window is owner.bridge


@pytest.mark.parametrize("transport", ["reply", "bridge"])
@pytest.mark.parametrize("then", ["verified document", "renderer swap"])
def test_a_posted_panic_stays_owed_across_a_document_that_got_no_ticket(native, tmp_path, transport, then):
    owner = renderer(tmp_path)
    view, nav = owner.view, owner.navigation
    nav.load_tagged(view, Mock())
    challenge = nav.challenge
    commit_document(view, nav)  # the verified document whose stop is pressed
    ticket = nav.ticket
    receiver, guard = launcher_webkit._native_classes()[1](), native.guard()
    vendor = owner.browser.js_bridge
    receiver.owner = guard.owner = owner
    guard.is_bridge, guard.original = True, vendor

    def posted(mark, *, main=True):
        incoming = message(view, nav, "", main=main)
        incoming.frameInfo().request().valueForHTTPHeaderField_.return_value = mark
        incoming.name.return_value = launcher_webkit.PANIC_HANDLER
        incoming.body.return_value = (ticket if transport == "reply" else
                                      json.dumps({"funcName": "panic_stop", "params": [ticket], "id": "press"}))
        return incoming

    def deliver(incoming):
        if transport == "bridge":
            return guard.userContentController_didReceiveScriptMessage_(None, incoming)
        reply = Mock()
        receiver.userContentController_didReceiveScriptMessage_replyHandler_(None, incoming, reply)
        return reply.call_args.args[0]

    gesture = posted(challenge)
    nav.webView_didCommitNavigation_(view, object())  # an intermediate document: a history step, no tagged response
    assert nav.ticket == ""
    owner.server_environment({})  # the restarted server may rotate its key before the next verified document
    owner.confirmed(object(), nav.port, replace=False)
    if then == "verified document":
        nav.load_tagged(view, Mock())
        commit_document(view, nav)
        assert nav.ticket not in ("", ticket)
    else:
        new, successor = candidate(owner, 9451)
        owner.receive(message(new, successor, "open", staged=True))
        successor.webView_didFinishNavigation_(new, successor.expected_navigation)
        ready(owner, new, successor)
        assert owner.view is new and owner.browser.webview is new
    with pytest.raises(ValueError, match="unverified"):  # the stop it is owed opens no room
        owner.receive(message(view, nav, "open", ticket=ticket))
    for unrelated in (posted("c" * 64), posted(None), posted(challenge, main=False)):
        assert deliver(unrelated) is None
    owner.background.request_panic.assert_not_called()
    result = deliver(gesture)
    if transport == "reply":
        assert result == {"status": "stopping"}
    deliver(gesture)  # owed once: a repeat is not a second stop
    owner.background.request_panic.assert_called_once_with("desktop_bridge")
    vendor.userContentController_didReceiveScriptMessage_.assert_not_called()


def test_a_verified_bridge_panic_is_the_launchers_before_the_vendor_bridge_answers_its_page(native, tmp_path):
    owner = renderer(tmp_path)
    view, nav, order = owner.view, owner.navigation, []
    guard, vendor = native.guard(), owner.browser.js_bridge
    guard.owner, guard.is_bridge, guard.original = owner, True, vendor
    owner.background.request_panic.side_effect = order.append
    vendor.userContentController_didReceiveScriptMessage_.side_effect = lambda *_: order.append("page answered")

    def press(ticket, **frame):
        event = message(view, nav, "", **frame)
        event.body.return_value = json.dumps({"funcName": "panic_stop", "params": [ticket], "id": "press"})
        return event

    for unverified in (press("another-document"), press(nav.ticket, main=False),
                       press(nav.ticket, url="http://127.0.0.1:9450/onboarding")):
        guard.userContentController_didReceiveScriptMessage_(None, unverified)
    assert order == []
    guard.userContentController_didReceiveScriptMessage_(None, press(nav.ticket))
    assert order == ["desktop_bridge", "page answered"]


@pytest.mark.parametrize("failing_control", ["chrome", "zoom"])
def test_replacement_releases_old_cocoa_view_if_a_window_control_fails(tmp_path, failing_control):
    owner = renderer(tmp_path)
    old = owner.view
    getattr(owner.background, failing_control).renderer_changed.side_effect = RuntimeError("control failed")
    new, nav = candidate(owner, 9451)
    owner.receive(message(new, nav, "open", staged=True))
    nav.webView_didFinishNavigation_(new, nav.expected_navigation)
    reply = ready(owner, new, nav)
    assert drained(owner, {})
    assert owner.view is new and owner.browser.webview is new and owner.candidate is None
    old.loadHTMLString_baseURL_.assert_called_once_with("", None)
    old.release.assert_called_once()
    reply.assert_called_once_with({}, None)


def test_vendor_finish_failure_restores_previous_view_and_discards_candidate(tmp_path):
    owner = renderer(tmp_path)
    old = owner.view
    new, nav = candidate(owner, 9451)
    owner.receive(message(new, nav, "open", staged=True))
    nav.previous.webView_didFinishNavigation_.side_effect = RuntimeError("didFinish failed")
    nav.webView_didFinishNavigation_(new, nav.expected_navigation)
    reply = ready(owner, new, nav)
    assert owner.browser.window.setContentView_.call_args_list == [call(new), call(old)]
    assert owner.view is old and owner.browser.webview is old and owner.candidate is None
    assert reply.call_count == 1 and reply.call_args.args[0] is None
    new.loadHTMLString_baseURL_.assert_called_once_with("", None)
    old.release.assert_not_called()


def test_older_builtin_waits_for_disposition_then_opens_without_receiving_old_text(tmp_path, native):
    owner = renderer(tmp_path)
    old, prior = owner.view, owner.navigation
    owner.session.apply([("ouro_chat_draft", "last words"), ("ouro_chat_input_history", "older words")])
    new, nav = candidate(owner, 9451)
    declined = Mock()
    owner.receive(message(new, nav, "open"), declined)
    nav.webView_didFinishNavigation_(new, nav.expected_navigation)
    new.callAsyncJavaScript_arguments_inFrame_inContentWorld_completionHandler_.call_args.args[-1](True, None)
    assert owner.browser.webview is old
    assert drained(owner, owner.session.items.copy())
    assert declined.call_args.args[0] is None
    assert owner.view is old and owner.waiting_for_builtin
    assert "ouro:renderer-incompatible" in old.evaluateJavaScript_completionHandler_.call_args.args[0]
    owner.receive(message(old, prior, "update", changes='[["ouro_chat_draft",null]]'))
    resume, generation = native.app.callAfter.call_args.args
    resume(generation)
    new, nav = owner.candidate, owner.candidate_nav
    commit_document(new, nav)
    reply = Mock()
    owner.receive(message(new, nav, "open"), reply)
    nav.webView_didFinishNavigation_(new, nav.expected_navigation)
    inspect = new.callAsyncJavaScript_arguments_inFrame_inContentWorld_completionHandler_.call_args.args[-1]
    reply.assert_not_called()
    assert owner.browser.webview is old
    inspect(True, None)
    assert drained(owner, {"ouro_chat_input_history": "older words"})
    assert owner.view is new and owner.session.items == {"ouro_chat_input_history": "older words"}
    reply.assert_called_once_with({}, None)


def test_uninitialized_older_builtin_cannot_take_over_an_empty_window(tmp_path):
    owner = renderer(tmp_path)
    old = owner.view
    new, nav = candidate(owner, 9451)
    nav.webView_didFinishNavigation_(new, nav.expected_navigation)
    reply = Mock()
    owner.receive(message(new, nav, "open"), reply)
    check = new.callAsyncJavaScript_arguments_inFrame_inContentWorld_completionHandler_.call_args.args[-1]
    check(False, None)
    assert owner.view is old and owner.candidate is None and not owner.requested
    assert reply.call_count == 1 and reply.call_args.args[0] is None


def test_no_interface_can_display_the_unconfirmed_input_notice_natively(tmp_path, monkeypatch):
    owner = renderer(tmp_path)
    owner.session.update("ouro_chat_draft", "still private")
    settings = tmp_path / "custom-settings.json"
    monkeypatch.setattr("ouroboros.config.SETTINGS_PATH", settings)
    alert = Mock()
    monkeypatch.setitem(sys.modules, "AppKit", SimpleNamespace(NSAlert=alert))
    new, nav = start_candidate(owner, 9451)
    commit_document(new, nav, "none")
    nav.webView_didFinishNavigation_(new, nav.expected_navigation)
    owner.finish_exchange(owner.exchange, confirmed=False)
    sheet = alert.alloc().init()
    sheet.beginSheetModalForWindow_completionHandler_.assert_called_once_with(owner.browser.window, None)
    guidance = sheet.setInformativeText_.call_args.args[0]
    assert "Settings › This computer › Advanced" in guidance
    assert "OUROBOROS_UI_DIR" in guidance and f"{settings} while Ouroboros is stopped" in guidance
    assert owner.view is new and owner.candidate is None
    assert owner.session.items == {"ouro_chat_draft": "still private"}
    with pytest.raises(ValueError):
        owner.receive(message(new, nav, "open", staged=True))
    restored, room = candidate(owner, 9452)
    owner.receive(message(restored, room, "open", staged=True))
    room.webView_didFinishNavigation_(restored, room.expected_navigation)
    ready(owner, restored, room).assert_called_once_with({"ouro_chat_draft": "still private"}, None)


def silent_editor(owner, native, port=9451, shown=False):
    """A ready successor behind a window whose known unsaved edits never get an answer by the collection deadline."""
    owner.session.edits.add("Files")
    new, nav = candidate(owner, port)
    owner.receive(message(new, nav, "open", staged=True))
    nav.webView_didFinishNavigation_(new, nav.expected_navigation)
    reply = ready(owner, new, nav)
    assert owner.browser.webview is not new  # the edited window stays on screen
    _, expire, *args = native.app.callLater.call_args.args
    expire(*args)
    return new, nav, reply, lambda: expire(*args)


@pytest.mark.parametrize("answer", ["open fresh", "keep editing"])
def test_a_silent_window_with_known_unsaved_edits_leaves_them_to_its_owner(tmp_path, native, monkeypatch, answer):
    """The collection deadline never discards a form the shell already knows is unsaved, and never leaves the
    window to a page that may be hung: the owner answers once on the shell's own sheet. Only "Open Fresh Window"
    gives the edits up, and the sheet names editors, never their text."""
    alert = Mock()
    monkeypatch.setitem(sys.modules, "AppKit", SimpleNamespace(NSAlert=alert))
    owner = renderer(tmp_path)
    old = owner.view
    new, nav, reply, expire = silent_editor(owner, native)
    assert owner.view is old and owner.candidate is new and owner.session.edits == {"Files"}
    reply.assert_not_called()
    old.loadHTMLString_baseURL_.assert_not_called()  # nothing installed and nothing discarded before the answer
    sheet = alert.alloc().init()
    assert sheet.setMessageText_.call_args.args[0] == "Your unsaved edits are still in this window"
    words = sheet.setInformativeText_.call_args.args[0]
    assert "did not confirm your edits in Files" in words and "give these edits up" in words
    assert [button.args[0] for button in sheet.addButtonWithTitle_.call_args_list] == ["Keep Editing", "Open Fresh Window"]
    window, decide = sheet.beginSheetModalForWindow_completionHandler_.call_args.args
    assert window is owner.browser.window
    timers = native.app.callLater.call_count
    expire()  # the deadline does not ask twice, and the question has no timer of its own
    assert sheet.beginSheetModalForWindow_completionHandler_.call_count == 1 and native.app.callLater.call_count == timers
    if answer == "open fresh":
        decide(1001)  # NSAlertSecondButtonReturn
        assert owner.view is new and owner.browser.webview is new and not owner.session.edits
        reply.assert_called_once_with({}, None)
        old.loadHTMLString_baseURL_.assert_called_once_with("", None)
        assert "ouro:room-input-unconfirmed" in new.evaluateJavaScript_completionHandler_.call_args.args[0]
    else:
        decide(1000)  # the default button
        assert owner.view is old and owner.candidate is None and owner.session.edits == {"Files"}
        assert owner.pending is nav.generation and not owner.requested
        assert reply.call_count == 1 and reply.call_args.args[0] is None
        old.loadHTMLString_baseURL_.assert_not_called()
        assert queued(native, owner._replace) == []
        owner.receive(message(old, owner.navigation, "editor", name="Files", dirty=False))
        assert queued(native, owner._replace) == [(nav.generation,)]  # finishing the edits asks for the fresh window
    installed = owner.view
    decide(1001)  # an answer that outlives its exchange gives nothing up
    assert owner.view is installed and owner.browser.window.setContentView_.call_count == (answer == "open fresh")


def test_an_unshown_question_gives_no_edits_up_and_a_late_answer_of_the_page_still_counts(tmp_path, native, monkeypatch):
    alert = Mock()
    alert.alloc.side_effect = RuntimeError("AppKit refused the sheet")
    monkeypatch.setitem(sys.modules, "AppKit", SimpleNamespace(NSAlert=alert))
    owner = renderer(tmp_path)
    old = owner.view
    new, nav, reply, _ = silent_editor(owner, native)
    assert owner.view is old and owner.candidate is None and owner.session.edits == {"Files"}
    assert owner.pending is nav.generation and not owner.requested  # held as if the owner chose to keep editing
    old.loadHTMLString_baseURL_.assert_not_called()

    sheet = Mock()
    monkeypatch.setitem(sys.modules, "AppKit", SimpleNamespace(NSAlert=sheet))
    new, nav, reply, _ = silent_editor(owner, native, 9452)  # asked this time
    decide = sheet.alloc().init().beginSheetModalForWindow_completionHandler_.call_args.args[1]
    assert drained(owner, {}, ["Files"])  # the page answers late, its edits still unsaved: the question stands
    assert owner.candidate is new and sheet.alloc().init().beginSheetModalForWindow_completionHandler_.call_count == 1
    timers = native.app.callLater.call_count
    assert drained(owner, {"ouro_chat_draft": "typed meanwhile"})  # then it saved them: the ordinary cut-over, on a
    assert owner.browser.webview is new and owner.view is old      # fresh deadline for its final turn
    assert native.app.callLater.call_count == timers + 1
    assert drained(owner, {"ouro_chat_draft": "typed meanwhile"})
    assert owner.view is new and not owner.session.edits
    reply.assert_called_once_with({"ouro_chat_draft": "typed meanwhile"}, None)
    decide(1000)  # the sheet was still up: its answer belongs to an exchange that has settled
    assert owner.view is new and owner.browser.webview is new


def turns(native):
    """Run the AppKit turns queued so far, each once."""
    waiting = list(native.app.callAfter.call_args_list)
    native.app.callAfter.reset_mock()
    for turn in waiting:
        turn.args[0](*turn.args[1:])


def ask_to_resume(owner, view, nav):
    receiver = launcher_webkit._native_classes()[1]()
    receiver.owner = owner
    reply = Mock()
    receiver.userContentController_didReceiveScriptMessage_replyHandler_(None, message(view, nav, "resume_queue"), reply)
    return reply


@pytest.mark.parametrize("word,answer", [("same port", True), ("another port", False), ("replacement", False)])
def test_a_reconnect_that_outruns_the_launchers_probe_is_answered_by_its_word_not_refused(native, tmp_path, word, answer):
    """After a crash the page can reach the restarted server before the launcher's health probe confirms it. The
    queue is neither released on a guess nor turned into held sends: the answer waits, with no timer, for the
    lifecycle owner's next word, and only the ordinary continuation at the same port lets the queue go on."""
    owner = renderer(tmp_path)
    view, nav = owner.view, owner.navigation
    alive = {owner.latest[:2]}
    owner.current = lambda proc, port: (proc, port) in alive
    ask_to_resume(owner, view, nav).assert_called_once_with(True, None)  # its own live server: at once

    alive.clear()  # that server crashed; nothing was requested
    timers = native.app.callLater.call_count
    waiting = ask_to_resume(owner, view, nav)
    waiting.assert_not_called()
    assert owner.receive(message(view, nav, "resume_queue")) is False  # a caller without a reply door still hears no
    owner.server_exited(owner.latest[0])
    turns(native)
    waiting.assert_not_called()
    assert native.app.callLater.call_count == timers

    successor = object()
    owner.server_environment({})
    if word == "replacement":
        owner.replacement_requested()
    else:
        port = nav.port + (word == "another port")
        alive.add((successor, port))
        owner.confirmed(successor, port, replace=False)
    waiting.assert_not_called()  # the word arrives on the lifecycle thread; AppKit answers in its own turn
    turns(native)
    waiting.assert_called_once_with(answer, None)
    turns(native)
    assert waiting.call_count == 1


def test_a_word_about_a_server_that_died_again_keeps_the_answer_waiting(native, tmp_path):
    owner = renderer(tmp_path)
    view, nav = owner.view, owner.navigation
    alive = set()
    owner.current = lambda proc, port: (proc, port) in alive
    waiting = ask_to_resume(owner, view, nav)
    first, second = object(), object()
    owner.confirmed(first, nav.port, replace=False)  # confirmed, then gone before AppKit's turn
    turns(native)
    waiting.assert_not_called()
    alive.add((second, nav.port))
    owner.confirmed(second, nav.port, replace=False)
    turns(native)
    waiting.assert_called_once_with(True, None)


HELP = "/onboarding?builtin=1"


def helps_page(view):
    """How often ``view`` was given Help's page: a plain request, where the shell's own root loads are tagged copies."""
    plain = sys.modules["Foundation"].NSURLRequest.requestWithURL_.return_value
    return sum(load.args[0] is plain for load in view.loadRequest_.call_args_list)


def help_addresses():
    return [made.args[0] for made in sys.modules["Foundation"].NSURL.URLWithString_.call_args_list if HELP in made.args[0]]


@pytest.mark.parametrize("outcome", ["installed", "reversed"])
def test_helps_page_waits_for_the_replacement_to_settle_and_loads_once_from_the_child_confirmed_then(
        native, tmp_path, outcome):
    """Help names a page, not an address: the port its menu read may belong to a child that is gone before AppKit loads
    anything. A view loaded while its successor is asked for or loading would lose the page to it, and a shown successor
    refuses every address but its root. So only the path is carried, and loaded once, in the view the window settles
    on, from the child that is live in that turn."""
    owner = renderer(tmp_path)
    old = owner.view
    new, nav = start_candidate(owner, 9451)  # asked for and loading off screen, at another port
    owner.load("http://127.0.0.1:9450" + HELP)  # the address the menu read: the server that is being replaced
    assert not old.loadRequest_.called  # nothing on the menu thread
    turns(native)
    assert not helps_page(old) and not helps_page(new)  # carried while the successor loads
    commit_document(new, nav)
    owner.receive(message(new, nav, "open", staged=True))
    nav.webView_didFinishNavigation_(new, nav.expected_navigation)
    ready(owner, new, nav)
    assert owner.browser.webview is new and owner.view is old and owner.exchange is not None
    turns(native)
    assert not helps_page(old) and not helps_page(new)  # and while it hands over
    if outcome == "installed":
        assert drained(owner, {})
        live, other = new, old
    else:
        owner.refuse("inert fixture reverses the presented successor")
        live, other = old, new
    assert owner.view is live and owner.browser.webview is live
    assert helps_page(live) == 1 and not helps_page(other)
    assert help_addresses() == ["http://127.0.0.1:9451" + HELP]  # the confirmed child's, never the address read at the menu
    owner.drop_candidate()
    turns(native)
    assert helps_page(live) == 1  # once


@pytest.mark.parametrize("editing", [False, True], ids=["clean", "unsaved form"])
def test_helps_page_never_replays_an_address_read_from_a_child_that_exited(native, tmp_path, editing):
    owner = renderer(tmp_path)
    old, prior = owner.view, owner.navigation
    alive = set()
    owner.current = lambda proc, port: (proc, port) in alive  # the child the menu read its port from has exited
    if editing:
        owner.receive(message(old, prior, "editor", name="Your rules", dirty=True))
    owner.load("http://127.0.0.1:9450" + HELP)
    turns(native)
    assert not helps_page(old) and help_addresses() == []
    successor = object()
    owner.server_environment({})
    alive.add((successor, 9460))
    owner.confirmed(successor, 9460, replace=False)  # an ordinary restart at another port: the view is not replaced
    turns(native)
    if editing:
        assert not helps_page(old) and owner.session.edits == {"Your rules"}  # a live child is no leave to navigate
        owner.receive(message(old, prior, "editor", name="Your rules", dirty=False))
    assert helps_page(old) == 1 and help_addresses() == ["http://127.0.0.1:9460" + HELP]
    owner.confirmed(successor, 9460, replace=False)
    turns(native)
    assert helps_page(old) == 1  # the page was owed once


def test_helps_page_opens_in_a_window_that_waits_for_room_input_and_waits_once_its_replacement_is_asked_again(
        native, tmp_path):
    """A window kept because the selected interface cannot take its room input tells its owner to use Help: there the
    page loads in the view that stayed. Once that replacement is asked for again the page waits for it, even in the
    turns before the successor exists."""
    owner = renderer(tmp_path)
    old = owner.view
    owner.session.update("ouro_chat_reply", "private quotation")
    new, nav = start_candidate(owner, 9451)
    commit_document(new, nav, "directory")
    nav.webView_didFinishNavigation_(new, nav.expected_navigation)
    assert drained(owner, {"ouro_chat_reply": "private quotation"})
    assert owner.view is old and owner.waiting_for_builtin and owner.pending is nav.generation and not owner.requested
    owner.load("http://127.0.0.1:9450" + HELP)
    turns(native)
    assert helps_page(old) == 1 and help_addresses() == ["http://127.0.0.1:9451" + HELP]
    owner.server_environment({})
    owner.confirmed(object(), 9452, replace=True)  # another Restart while the window still waited for its room input
    assert owner.requested and owner.waiting_for_builtin and owner.candidate is None
    owner.load("http://127.0.0.1:9452" + HELP)  # Help again, in a turn before the successor's own
    owner.schedule_confirmed()
    turns(native)
    again = owner.candidate
    assert again is not None and owner.candidate_nav.port == 9452
    assert helps_page(old) == 1 and not helps_page(again)  # carried: the view that stayed is being replaced


@pytest.mark.parametrize("replacing", [False, True], ids=["no replacement", "held replacement"])
def test_helps_page_never_takes_the_place_of_a_document_with_unsaved_forms(native, tmp_path, monkeypatch, replacing):
    """Loading Help's page navigates the document away, and with it the forms the shell knows are unsaved. The page
    waits for them instead and the shell says so. An exchange reversed to keep those forms does not then load the page
    over them either: it follows the owner's last edit, into the fresh window those edits were holding."""
    alert = Mock()
    monkeypatch.setitem(sys.modules, "AppKit", SimpleNamespace(NSAlert=alert))
    owner = renderer(tmp_path)
    old, prior = owner.view, owner.navigation
    owner.receive(message(old, prior, "editor", name="Files", dirty=True))
    if replacing:
        new, nav = candidate(owner, 9451)
        owner.receive(message(new, nav, "open", staged=True))
        nav.webView_didFinishNavigation_(new, nav.expected_navigation)
        ready(owner, new, nav)
        assert owner.browser.webview is old and owner.exchange is not None  # held behind the edited window
    owner.load("http://127.0.0.1:9450" + HELP)
    turns(native)
    if replacing:
        drained(owner, {}, ["Files"])  # the window keeps its edits: the exchange is reversed
        assert owner.candidate is None and owner.pending is nav.generation and not owner.requested
    sheet = alert.alloc().init()
    sheet.beginSheetModalForWindow_completionHandler_.assert_called_once_with(owner.browser.window, None)
    words = sheet.setInformativeText_.call_args.args[0]
    assert "your edits in Files" in words and "Nothing was saved or discarded" in words
    assert ("the fresh window opens by itself" if replacing else "the page you chose in Help opens by itself") in words
    assert not helps_page(old) and owner.session.edits == {"Files"}
    old.loadHTMLString_baseURL_.assert_not_called()
    turns(native)  # a later word of the lifecycle owner changes nothing and says nothing twice
    assert not helps_page(old) and sheet.beginSheetModalForWindow_completionHandler_.call_count == 1
    owner.receive(message(old, prior, "editor", name="Files", dirty=False))
    if not replacing:
        assert helps_page(old) == 1 and help_addresses() == ["http://127.0.0.1:9450" + HELP]
        return
    owner._word()  # a word of the lifecycle owner that AppKit hears before the fresh window's own turn
    assert not helps_page(old)  # the fresh window is asked for again first
    turns(native)
    new, nav = candidate_started(owner)
    owner.receive(message(new, nav, "open", staged=True))
    nav.webView_didFinishNavigation_(new, nav.expected_navigation)
    ready(owner, new, nav)
    assert not helps_page(new)
    assert drained(owner, {})
    assert owner.view is new and helps_page(new) == 1 and not helps_page(old)
    assert help_addresses() == ["http://127.0.0.1:9451" + HELP]


def candidate_started(owner):
    """The successor an AppKit turn started by itself, with its verified document committed."""
    view, nav = owner.candidate, owner.candidate_nav
    commit_document(view, nav)
    return view, nav


def shown(tmp_path, monkeypatch, alive):
    """A main window as the launcher makes it, up to pywebview's ``before_show``: the turn before its first load."""
    owner = MainRenderer(tmp_path, lambda proc, port: (proc, port) in alive)
    window, browser = MagicMock(name="pywebview window"), Mock(name="BrowserView")
    window.gui.__name__, window._callbacks, window.uid = "webview.platforms.cocoa", {}, "master"
    window.real_url = "http://127.0.0.1:9450"  # from the child that was confirmed when the launcher made the window
    browser.pywebview_window = window
    sys.modules["webview.platforms.cocoa"].BrowserView.instances = {"master": browser}
    monkeypatch.setitem(sys.modules, "AppKit", SimpleNamespace(NSWindowWillCloseNotification="will close"))
    monkeypatch.setattr(sys.modules["Foundation"], "NSNotificationCenter", Mock(), raising=False)
    owner.attach(window, Mock(name="background"))
    return owner, window


def child(owner, alive, port, *, replace=False):
    """The lifecycle owner's word about a new child of its own, serving at ``port``."""
    proc = object()
    alive.clear()
    owner.server_environment({})
    alive.add((proc, port))
    owner.confirmed(proc, port, replace=replace)
    return proc


def decided(nav, view, act):
    handler = Mock()
    handler.__block_signature__ = True
    nav.webView_decidePolicyForNavigationAction_decisionHandler_(view, act, handler)
    return handler


@pytest.mark.parametrize("restart", [False, True], ids=["crash restart", "requested restart"])
def test_the_first_document_is_bound_when_it_loads_not_when_the_window_was_made(native, tmp_path, monkeypatch, restart):
    """Between making the window and its first navigation the initial child can exit and its successor take another
    port. The address the window was made with then names a server that is gone: loading it leaves the window empty
    with nothing to recover it. pywebview's request is only the cue: the first document is this shell's own tagged
    request, bound in the turn that makes it to the child that is live then."""
    alive = set()
    owner, window = shown(tmp_path, monkeypatch, alive)
    child(owner, alive, 9450)  # confirmed when the window was made
    child(owner, alive, 9460, replace=restart)  # it exited; its successor is the launcher's live child now
    owner._before_show()
    view, nav = owner.view, owner.navigation
    allowed = nav.previous.webView_decidePolicyForNavigationAction_decisionHandler_
    assert window.real_url == "http://127.0.0.1:9460/" and owner.first and owner.installed
    # WebKit keeps only the last request it is given, and pywebview's own follows this turn: none is made in it.
    view.loadRequest_.assert_not_called()
    assert queued(native, owner._replace) == ([(owner.latest,)] if restart else [])
    asked = decided(nav, view, action("about:blank", "http://127.0.0.1:9450/"))  # the address the window was made with
    asked.assert_called_once_with(native.kit.WKNavigationActionPolicyCancel)
    allowed.assert_not_called()  # never loaded, tagged or not
    assert view.loadRequest_.call_count == 1 and nav.challenge  # answered by the live child's root, under a challenge
    assert sys.modules["Foundation"].NSURL.URLWithString_.call_args.args[0] == "http://127.0.0.1:9460/"
    assert (nav.port, nav.key) == owner.latest[1:] and owner.presented is owner.latest
    assert owner.requested is restart  # a requested restart is still owed in this turn, and this is its document:
    own = action("about:blank", "http://127.0.0.1:9460/", tag=nav.challenge)
    decided(nav, view, own).assert_not_called()  # the shell's own request goes on (a page's reload would wait)
    assert allowed.call_args.args[:2] == (view, own) and allowed.call_count == 1
    turns(native)  # the lifecycle owner's words, heard by AppKit only now: that request is already on its way
    assert view.loadRequest_.call_count == 1 and owner.candidate is None
    assert owner.pending is None and not owner.requested  # a requested restart found no document to replace
    decided(nav, view, action("about:blank", "http://127.0.0.1:9460/"))  # the same address again, untagged
    assert view.loadRequest_.call_count == 1 and allowed.call_count == 1  # cancelled: the tagged request stays alone
    owner.load("http://127.0.0.1:9450" + HELP)  # Help, before the window has its document: any request but that
    turns(native)                               # root would be cancelled there, so the page waits for it
    assert not helps_page(view)
    commit_document(view, nav)
    assert not owner.first and nav.ticket and nav.generation is owner.latest
    assert helps_page(view) == 1 and help_addresses() == ["http://127.0.0.1:9460" + HELP]
    other = action("http://127.0.0.1:9460/", "http://127.0.0.1:9460/onboarding")
    decided(nav, view, other)  # once it has a document the window navigates as before
    assert allowed.call_args.args[:2] == (view, other) and allowed.call_count == 2


def test_a_window_without_a_live_child_loads_nothing_until_the_lifecycle_owner_speaks_again(
        native, tmp_path, monkeypatch, startup_clock):
    alive = set()
    owner, window = shown(tmp_path, monkeypatch, alive)
    first = child(owner, alive, 9450)
    alive.clear()  # the initial child exited before the first navigation; no successor is confirmed yet
    owner._before_show()
    view, nav = owner.view, owner.navigation
    loads, timers = (lambda: view.loadRequest_.call_count), (lambda: native.app.callLater.call_count)

    def fail():
        nav.webView_didFailProvisionalNavigation_withError_(view, nav.expected_navigation, Mock())

    assert owner.first and nav.port is None and owner.presented is None
    assert window.real_url == "http://127.0.0.1:9450"  # no child is live: the address is left as it was made
    decided(nav, view, action("about:blank", "http://127.0.0.1:9450/")).assert_called_once_with(
        native.kit.WKNavigationActionPolicyCancel)
    nav.previous.webView_decidePolicyForNavigationAction_decisionHandler_.assert_not_called()
    turns(native)
    assert loads() == 0  # the exited child's address is not loaded, and nothing is asked of it again
    assert not owner.server_exited(first)
    child(owner, alive, 9461)  # an ordinary restart: no replacement was asked, but the window still has no document
    assert loads() == 0  # the word arrives on the lifecycle thread; AppKit loads in its own turn
    turns(native)
    assert loads() == 1 and nav.port == 9461 and window.real_url == "http://127.0.0.1:9461/"
    assert owner.presented is owner.latest
    child(owner, alive, 9462)  # that child exits with the load still on its way: the next one takes its place
    turns(native)
    assert loads() == 2 and (nav.port, nav.key) == owner.latest[1:] and owner.presented is owner.latest
    # WebKit's own word that this load failed asks again after a pause, a bounded number of times, and only while
    # that child is still the live one. Nothing else is timed or polled.
    elapsed, before = 0, timers()
    for attempt, wait in enumerate(launcher_webkit._RETRY_SECONDS, start=1):
        fail()
        assert timers() == before + attempt and loads() == 1 + attempt and nav.port is None  # owed, not yet asked
        elapsed += wait
        startup_clock(elapsed)
        assert loads() == 2 + attempt and nav.port == 9462
    before, loaded = timers(), loads()
    fail()
    assert timers() == before and loads() == loaded and owner.first and nav.port is None  # no further attempt
    child(owner, alive, 9463)  # only the lifecycle owner's next word loads it
    turns(native)
    assert loads() == loaded + 1 and nav.port == 9463
    alive.clear()  # that child died with the load on its way: its failure is the lifecycle owner's to follow
    fail()
    assert timers() == before and loads() == loaded + 1
    child(owner, alive, 9464, replace=True)  # a requested restart finds no document to replace: this is the first one
    owner.schedule_confirmed()
    turns(native)
    assert loads() == loaded + 2 and nav.port == 9464 and owner.candidate is None
    assert owner.pending is None and not owner.requested
    commit_document(view, nav)
    assert not owner.first and nav.ticket
    fail()
    assert timers() == before  # a window that has a document keeps it: nothing is asked again


def test_a_failed_successor_load_is_asked_again_a_bounded_number_of_times_while_its_child_is_live(
        native, tmp_path, startup_clock):
    """The old page of a server that moved to another port can never ask for the new one, so after WebKit's own word
    that the successor's load failed nothing else would retry. The shell asks again itself, after a pause and a bounded
    number of times, then refuses as before."""
    owner = renderer(tmp_path)
    old = owner.view
    view, nav = start_candidate(owner, 9451)
    generation, elapsed = nav.generation, 0
    nav.webView_didFailProvisionalNavigation_withError_(view, object(), Mock())  # not the load this shell issued
    assert owner.candidate is view
    for wait in launcher_webkit._RETRY_SECONDS:
        nav.webView_didFailProvisionalNavigation_withError_(view, nav.expected_navigation, Mock())
        assert owner.candidate is None and owner.pending is generation and owner.requested  # still owed, and quietly
        old.evaluateJavaScript_completionHandler_.assert_not_called()
        view.loadHTMLString_baseURL_.assert_called_once_with("", None)
        startup_clock(elapsed + wait - 0.5)
        assert owner.candidate is None
        elapsed += wait
        startup_clock(elapsed)
        view, nav = owner.candidate, owner.candidate_nav
        assert nav.generation is generation and nav.port == 9451 and owner.browser.webview is old
    nav.webView_didFailProvisionalNavigation_withError_(view, nav.expected_navigation, Mock())
    assert owner.candidate is None and owner.pending is None and not owner.requested
    assert "ouro:renderer-refused" in old.evaluateJavaScript_completionHandler_.call_args.args[0]
    startup_clock(elapsed + 60)  # no timer, earlier candidate or retry is left to start another
    assert owner.candidate is None and owner.view is old


@pytest.mark.parametrize("stage", ["provisional", "committed"])
def test_a_successor_whose_child_is_gone_is_left_to_the_lifecycle_owner(native, tmp_path, stage):
    """WebKit can report the failed load before the lifecycle owner has seen that child exit. Refusing then would
    forget the replacement the owner still owes: its next child would be confirmed as an ordinary continuation and the
    window would keep a page whose server is gone."""
    owner = renderer(tmp_path)
    old = owner.view
    view, nav = start_candidate(owner, 9451)
    owner.current = lambda proc, port: False  # exited; its owner has not said so yet
    if stage == "committed":
        nav.committed_navigation = nav.expected_navigation
        nav.webView_didFailNavigation_withError_(view, nav.expected_navigation, Mock())
    else:
        nav.webView_didFailProvisionalNavigation_withError_(view, nav.expected_navigation, Mock())
    assert owner.candidate is None and owner.pending is nav.generation and owner.requested
    native.app.callLater.assert_called_once()  # the dropped candidate's startup deadline only: no retry was scheduled
    old.evaluateJavaScript_completionHandler_.assert_not_called()  # and no refusal was announced
    assert owner.server_exited(nav.generation[0]) is True  # the replacement is still owed to the next child


def test_a_retry_owed_to_an_earlier_generation_cannot_touch_a_later_successor(native, tmp_path, startup_clock):
    owner = renderer(tmp_path)
    view, nav = start_candidate(owner, 9451)
    nav.webView_didFailProvisionalNavigation_withError_(view, nav.expected_navigation, Mock())  # asked again in a while
    later, successor = start_candidate(owner, 9452)  # the lifecycle owner confirmed another child before that
    startup_clock(launcher_webkit._RETRY_SECONDS[0])
    assert owner.candidate is later and owner.pending is successor.generation and owner.requested
    later.loadHTMLString_baseURL_.assert_not_called()
    assert later.loadRequest_.call_count == 1  # its own request only: the earlier generation's retry started nothing


def test_a_committed_successor_document_that_fails_is_refused_not_asked_again(native, tmp_path):
    owner = renderer(tmp_path)
    old = owner.view
    view, nav = candidate(owner, 9451)
    timers = native.app.callLater.call_count
    nav.webView_didFailNavigation_withError_(view, nav.expected_navigation, Mock())
    assert owner.candidate is None and owner.pending is None and not owner.requested
    assert native.app.callLater.call_count == timers
    assert "ouro:renderer-refused" in old.evaluateJavaScript_completionHandler_.call_args.args[0]


@pytest.mark.parametrize("held", [False, True], ids=["handing over", "held for unsaved forms"])
def test_a_previous_view_that_navigates_during_its_exchange_cannot_discard_the_successor(native, tmp_path, held):
    """A document that leaves the outgoing view can no longer confirm its last input or own its forms. It ends its own
    part of the exchange: the successor is kept and told that the last input is unconfirmed."""
    owner = renderer(tmp_path)
    old, prior = owner.view, owner.navigation
    owner.session.update("ouro_chat_draft", "last delivered")
    if held:
        owner.session.edits.add("Files")
    new, nav = candidate(owner, 9451)
    owner.receive(message(new, nav, "open", staged=True))
    nav.webView_didFinishNavigation_(new, nav.expected_navigation)
    reply = ready(owner, new, nav)
    assert owner.exchange is not None and owner.view is old and (owner.browser.webview is old) is held
    prior.webView_didCommitNavigation_(old, object())  # a history step or a link took the outgoing view elsewhere
    assert owner.view is new and owner.browser.webview is new and owner.candidate is None
    assert owner.pending is None and not owner.requested and not owner.session.edits
    reply.assert_called_once_with({"ouro_chat_draft": "last delivered"}, None)
    old.loadHTMLString_baseURL_.assert_called_once_with("", None)
    new.loadHTMLString_baseURL_.assert_not_called()
    assert "ouro:room-input-unconfirmed" in new.evaluateJavaScript_completionHandler_.call_args.args[0]


def test_a_tagged_request_overtaken_by_a_later_one_is_cancelled_without_tagging_a_third(native, tmp_path):
    """Two requests can be on their way at once, each asked about in its own turn. Tagging again at the turn of the
    earlier one would void the challenge of the later, whose turn would tag again in its place, without end."""
    owner = renderer(tmp_path)
    view, nav, root = owner.view, owner.navigation, "http://127.0.0.1:9450/"
    nav.previous = Mock()
    allowed = nav.previous.webView_decidePolicyForNavigationAction_decisionHandler_
    nav.load_tagged(view, Mock())
    first = nav.challenge
    decided(nav, view, action(root, root + "?page=reload"))  # the page asks for its root while that one is on its way
    second = nav.challenge
    assert second != first and view.loadRequest_.call_count == 2
    overtaken = decided(nav, view, action(root, root, tag=first))
    overtaken.assert_called_once_with(native.kit.WKNavigationActionPolicyCancel)
    assert view.loadRequest_.call_count == 2 and nav.challenge == second  # cancelled, and nothing tagged in its place
    allowed.assert_not_called()
    current = action(root, root, tag=second)
    decided(nav, view, current).assert_not_called()
    allowed.assert_called_once()
    assert allowed.call_args.args[:2] == (view, current)
    decided(nav, view, action(root, root, tag=first))  # a reload replays an old tag with nothing on its way: tagged anew
    assert view.loadRequest_.call_count == 3 and nav.challenge not in (first, second)


@pytest.mark.parametrize("source", ["builtin", "directory", "unmarked"])
def test_a_stop_without_a_ticket_counts_only_from_the_identified_main_document_that_got_none(native, tmp_path, source):
    """An older or a custom interface calls the bridge's stop with no argument. That call is the launcher's only from
    the installed view's main frame on the root this shell loaded and identified there, and it is taken in the turn it
    arrives, before the vendor bridge's worker. A stamped document proves itself by its ticket instead."""
    owner = renderer(tmp_path)  # its background is a Mock: no live stop, server or process owner
    view, nav, order = owner.view, owner.navigation, []
    alive = {owner.latest[:2]}
    owner.current = lambda proc, port: (proc, port) in alive
    guard, vendor = native.guard(), owner.browser.js_bridge
    guard.owner, guard.is_bridge, guard.original = owner, True, vendor
    owner.background.request_panic.side_effect = order.append
    vendor.userContentController_didReceiveScriptMessage_.side_effect = lambda *_: order.append("page answered")

    def press(params=(), **frame):
        event = message(view, nav, "", **frame)
        event.frameInfo().request().valueForHTTPHeaderField_.return_value = None
        event.body.return_value = json.dumps({"funcName": "panic_stop", "params": params if params is None else list(params),
                                              "id": "press"})
        return event

    def load(signed):
        nav.load_tagged(view, Mock())
        response_policy(view, nav, source="directory" if source == "directory" else "builtin", signed=signed)
        nav.webView_didCommitNavigation_(view, nav.expected_navigation)

    if source == "unmarked":
        alive.clear()
        load(signed=False)  # an unstamped root while no child of this launcher holds that port: nobody's document
        assert nav.source == "" and nav.ticket == ""
        guard.userContentController_didReceiveScriptMessage_(None, press())
        alive.add(owner.latest[:2])
        load(signed=False)  # an older server of this launcher: unstamped, identified by its own live child's port
    elif source == "directory":
        load(signed=True)  # a custom interface: stamped as selected, given no room ticket
    else:
        nav.source, nav.ticket = "builtin", ""  # an older built-in interface a replacement identified
    assert nav.source == source and nav.ticket == ""
    elsewhere = press()
    elsewhere.webView.return_value = Mock(name="another view")
    for refused in (press(main=False), press(url="http://127.0.0.1:9450/onboarding"), press(url="http://127.0.0.1:9451/"),
                    press(["a-ticket"]), press([None]), press(None), elsewhere):
        guard.userContentController_didReceiveScriptMessage_(None, refused)
    nav.ticket = "stamped"  # a document that holds a ticket never stops without it
    guard.userContentController_didReceiveScriptMessage_(None, press())
    nav.ticket, nav.source = "", ""  # and one this shell did not identify never stops at all
    guard.userContentController_didReceiveScriptMessage_(None, press())
    assert order == []
    nav.source = source
    guard.userContentController_didReceiveScriptMessage_(None, press())
    assert order == ["desktop_bridge", "page answered"]  # the launcher's stop first; the vendor call only answers


def test_a_registry_answer_counts_only_while_the_child_that_gave_it_is_still_confirmed(native, tmp_path, monkeypatch):
    owner = renderer(tmp_path)
    owner.session.update("ouro_chat_draft:7", "typed in Seven")
    alive = set()
    owner.current = lambda proc, port: (proc, port) in alive
    monkeypatch.setattr(launcher_webkit, "absent_rooms", lambda port, rooms: {"7"})
    gone, kept = object(), object()
    owner.confirmed(gone, 9451, replace=True)  # it exited while its registry was read
    assert owner.session.gone == set() and owner.session.has_room_input
    alive.add((kept, 9452))
    owner.confirmed(kept, 9452, replace=True)
    assert owner.session.gone == {"7"}


@pytest.mark.parametrize("source", ["directory", "none"])
@pytest.mark.parametrize("also", [{}, {"ouro_chat_draft": "still in Main"}, {"ouro_chat_files:8": "2"}],
                         ids=["only-the-lost-room", "main-draft-too", "another-room-too"])
def test_a_room_the_registry_no_longer_has_does_not_hold_the_interface_choice(native, tmp_path, monkeypatch, source, also):
    """A deleted Project's composer is gone, so its slots cannot be cleared there and the advice to clear them could
    never be followed. Once the confirmed server's registry positively no longer has the room, its slots stop holding
    an interface that cannot restore them. They stay in the shell until Quit and reach no other page; input the owner
    can still clear holds the interface exactly as before."""
    owner = renderer(tmp_path)
    old = owner.view
    lost = {"ouro_chat_draft:7": "typed in a Project deleted since", "ouro_chat_reply:7": "quote"}
    slots = {**lost, **also}
    owner.session.apply(list(slots.items()))
    asked = []
    monkeypatch.setattr(launcher_webkit, "absent_rooms", lambda port, rooms: asked.append((port, set(rooms))) or {"7"})
    new, nav = start_candidate(owner, 9451)
    assert asked == [(9451, {"7", "8"} if "ouro_chat_files:8" in also else {"7"})]  # room numbers only
    commit_document(new, nav, source)
    nav.webView_didFinishNavigation_(new, nav.expected_navigation)
    assert (owner.browser.webview is new) is (not also)  # nothing the owner could still clear: no window is held
    assert drained(owner, slots)  # the old page's final word still carries the lost room's slots, unchanged
    assert owner.session.items == slots and owner.session.gone == {"7"}
    notices = [notice.args[0] for notice in old.evaluateJavaScript_completionHandler_.call_args_list]
    if also:
        assert owner.view is old and owner.waiting_for_builtin and "ouro:renderer-incompatible" in notices[-1]
        return
    assert owner.view is new and not owner.waiting_for_builtin
    assert not any("ouro:renderer-incompatible" in notice for notice in notices)
    with pytest.raises(ValueError):  # the selected page has no room ticket: it is handed nothing
        owner.receive(message(new, nav, "open", staged=True))
    restored, room = candidate(owner, 9452)  # the built-in interface again: the shell kept every slot quietly
    owner.receive(message(restored, room, "open", staged=True))
    room.webView_didFinishNavigation_(restored, room.expected_navigation)
    ready(owner, restored, room).assert_called_once_with(slots, None)


def test_registration_failure_does_not_veto_a_compatible_source_handoff(tmp_path):
    owner = MainRenderer(tmp_path, lambda proc, port: True)
    assert not owner.installed
    with owner.prepare_exec(False, transfer=False) as (validate, inherit):
        validate()
        with inherit({}):
            pass


def test_source_handoff_checks_a_last_snapshot_and_keeps_the_shell_if_input_continues(tmp_path, native, monkeypatch):
    owner = renderer(tmp_path)
    ns = Mock()
    masks = ("KeyDown", "KeyUp", "LeftMouseDown", "LeftMouseUp", "RightMouseDown", "RightMouseUp",
             "OtherMouseDown", "OtherMouseUp")
    monkeypatch.setitem(sys.modules, "AppKit", SimpleNamespace(NSEvent=ns, **{
        "NSEventMask" + mask: 1 << i for i, mask in enumerate(masks)}))
    monkeypatch.setattr(launcher_webkit, "current_thread", object)  # stand-in for the lifecycle worker
    native.app.callAfter.side_effect = lambda fn: fn()

    def collect(_script, _completion):
        return owner.receive(message(owner.view, owner.navigation, "drained", handoff=owner.source_drain.token,
                                     items='{"ouro_chat_session_id":"last"}', editors=[]))

    owner.view.evaluateJavaScript_completionHandler_.side_effect = collect
    with owner.prepare_exec(False, transfer=False) as (validate, _):
        validate()
        assert owner.session.items == {"ouro_chat_session_id": "last"}
        monitor = ns.addLocalMonitorForEventsMatchingMask_handler_.call_args.args[1]
        press = object()
        assert monitor(press) is press  # the press still reaches its ordinary target
        with pytest.raises(launcher_webkit.SessionRefused, match="input continued"):
            validate()
    ns.removeMonitor_.assert_called_once()
    assert owner.source_drain is None

    monkeypatch.setattr(launcher_webkit, "_DRAIN_SECONDS", 0)
    owner.view.evaluateJavaScript_completionHandler_.side_effect = None
    with pytest.raises(launcher_webkit.SessionRefused, match="did not confirm"):
        with owner.prepare_exec(False, transfer=False):
            pytest.fail("a silent page cannot authorize replacing the source process")
    assert owner.source_drain is None and owner.view is owner.browser.webview

"""Ownership on the normal AppKit dispatch path, with stand-ins for the window and both views."""
from __future__ import annotations

import json
import sys
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock, call

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


def test_stale_generation_or_unsaved_editor_leaves_old_input_in_place(tmp_path):
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


def action(current, target, kind=0):
    frame, request, act = Mock(), Mock(), Mock()
    frame.request.return_value.URL.return_value.absoluteString.return_value = current
    frame.isMainFrame.return_value = True
    act.targetFrame.return_value, act.request.return_value = frame, request
    request.HTTPMethod.return_value = "GET"
    request.URL.return_value.absoluteString.return_value = target
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
    native.app.callAfter.assert_called_once()
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
    from ouroboros.launcher_background import DesktopApi

    owner = renderer(tmp_path)
    old, prior, old_bridge = owner.view, owner.navigation, owner.bridge
    challenge = "b" * 64
    prior.ticket = document_ticket(prior.key, challenge)
    guard, vendor = native.guard(), owner.browser.js_bridge
    guard.owner, guard.is_bridge, guard.original = owner, True, vendor
    api = DesktopApi()
    api._background = owner.background  # a Mock; no live stop, server, or process owner
    captured = []

    def dispatch(_, msg):
        call_body = json.loads(msg.body())
        captured.append(vendor.window)
        assert call_body["funcName"] == "panic_stop"
        return api.panic_stop(*call_body["params"])

    vendor.userContentController_didReceiveScriptMessage_.side_effect = dispatch

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
    assert captured == [old_bridge]
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

    def dispatch(*_):
        assert vendor.window is old_bridge
        owner.background.request_panic("desktop_bridge")

    vendor.userContentController_didReceiveScriptMessage_.side_effect = dispatch

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
        vendor.userContentController_didReceiveScriptMessage_.assert_not_called()
    deliver(gesture)
    owner.background.request_panic.assert_called_once_with("desktop_bridge")
    assert not old_bridge.active and vendor.window is owner.bridge


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


def test_a_silent_outgoing_editor_cannot_hold_a_ready_successor_forever(tmp_path, native):
    owner = renderer(tmp_path)
    old = owner.view
    owner.session.edits.add("Files")
    new, nav = candidate(owner, 9451)
    owner.receive(message(new, nav, "open", staged=True))
    nav.webView_didFinishNavigation_(new, nav.expected_navigation)
    reply = ready(owner, new, nav)
    assert owner.browser.webview is old
    _, expire, *args = native.app.callLater.call_args.args
    expire(*args)
    assert owner.view is new and not owner.session.edits
    reply.assert_called_once_with({}, None)
    assert "ouro:room-input-unconfirmed" in new.evaluateJavaScript_completionHandler_.call_args.args[0]


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

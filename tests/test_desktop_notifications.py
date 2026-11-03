"""System notifications from the desktop app: the bridge, the click route and each platform adapter.

Owner decisions 1A (the OS owns the sound), 2A (sent while the window is in front) and 3A (an
older app falls back and says so) — DESIGN §9. Every operating-system surface here is a stand-in:
no test asks for a permission, shows a banner or plays a sound. The Windows icon's pump runs for real
on a Python thread whose stand-in WinForms ``Application.Run`` is a message loop; only the shell's
own balloon call is replaced. The one real-PyObjC check loads the UserNotifications framework and builds
objects, but never touches the notification center.
"""
from __future__ import annotations

import ctypes
import inspect
import json
import pathlib
import queue
import re
import subprocess
import sys
import threading
import time
import types
from types import SimpleNamespace

import pytest

from ouroboros import desktop_notifications as dn
from ouroboros import launcher_background as lb
from ouroboros import launcher_tray

REPO = pathlib.Path(__file__).resolve().parents[1]


def wait_for(predicate, timeout=5.0):
    done = threading.Event()
    for _ in range(int(timeout / 0.01)):
        if predicate():
            return
        done.wait(0.01)
    raise AssertionError("condition not reached")


class Recorder(dn.NativeNotifier):
    platform = "test"

    def __init__(self, status="authorized", deliver=None, on_click=None):
        super().__init__(on_click or (lambda token: None))
        self.answer, self.deliver_with, self.calls = status, deliver, []

    def _status(self):
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer

    def _deliver(self, title, body, sound, token):
        self.calls.append((title, body, sound, token))
        if self.deliver_with is not None:
            return self.deliver_with()
        return dn.submitted(self.platform, "os" if sound else "off")


# ------------------------------------------------------------------------------- the bridge ---

def bridge(notifier):
    class Api(lb.DesktopApi):
        _background = SimpleNamespace(notifications=notifier, attention=lambda *a: {"ok": True, "args": a})

    return Api()


def pywebview_exposed(api):
    """pywebview 5.4 ``inject_pywebview.get_functions``: public bound methods of the js_api object."""
    return sorted(name for name in dir(api) if not name.startswith("_") and inspect.ismethod(getattr(api, name)))


def test_the_bridge_exposes_the_alert_methods_and_hides_its_background():
    api = bridge(Recorder())
    assert pywebview_exposed(api) == ["notify_owner", "panic_stop", "request_attention", "request_native_notifications",
                                      "set_interface_size", "set_native_appearance", "shell_info",
                                      "show_native_notification"]
    assert api.notify_owner(True, "Task finished", "", False) == {"ok": True, "args": (True, "Task finished", "", False)}


def test_launcher_main_api_inherits_the_bridge():
    source = (REPO / "launcher.py").read_text(encoding="utf-8")
    main_api = source[source.index("class MainApi("):]
    assert main_api.startswith("class MainApi(DesktopApi):"), "the frozen launcher must expose the bridge"
    assert "_background = background" in main_api.split("def ", 1)[0]
    assert "def request_attention" not in main_api, "one definition: launcher_background.DesktopApi"


def test_shell_info_is_the_apps_own_facts():
    info = bridge(Recorder(status="not_determined")).shell_info()
    assert info["shell_version"] == (REPO / "VERSION").read_text(encoding="utf-8").strip()
    assert info["persistent_storage"] is True
    assert info["native_notifications"] == {"available": True, "status": "not_determined", "platform": "test",
                                            "reason": ""}


def test_the_persistent_storage_claim_is_the_webview_flag():
    """``shell_info`` says the WebView keeps website data; that is true only while every window asks for it."""
    assert lb.PERSISTENT_WEBVIEW_STORAGE is True
    for rel in ("launcher.py", "ouroboros/launcher_onboarding.py"):
        calls = re.findall(r"^\s*webview\.start\((.*)\)\s*$", (REPO / rel).read_text(encoding="utf-8"), re.M)
        assert calls and all("private_mode=False" in args for args in calls), rel


def test_no_adapter_is_a_typed_unavailable_answer():
    api = bridge(None)
    assert api.shell_info()["native_notifications"]["status"] == "unavailable"
    assert api.shell_info()["interface_size"] == {"supported": False}, "no window zoom: the page keeps 100 % and says so"
    assert api.request_native_notifications() == {"available": False, "status": "unavailable",
                                                  "platform": dn.platform_name(), "reason": "no_platform_adapter"}
    assert api.show_native_notification("Task finished", "", True, "n1-a") == {
        "ok": False, "status": "unavailable", "platform": dn.platform_name(), "reason": "no_platform_adapter"}


def test_inputs_are_cleaned_and_failures_are_typed():
    notifier = Recorder()
    api = bridge(notifier)
    assert api.show_native_notification("  Task\nfinished ", "", True, "n4-k")["status"] == "submitted"
    assert notifier.calls[-1] == ("Task finished", "", True, "n4-k")
    api.show_native_notification("", "x" * 1000, False, "window.alert(1)")
    title, body, sound, token = notifier.calls[-1]
    assert title == "Ouroboros" and len(body) == 400 and sound is False
    assert re.fullmatch(r"n\d+-\d+", token), "a token that is not the page's shape is replaced, never echoed"

    notifier.deliver_with = lambda: (_ for _ in ()).throw(RuntimeError("boom"))
    assert api.show_native_notification("t", "", True, "n5-k") == {
        "ok": False, "status": "failed", "platform": "test", "reason": "RuntimeError: boom"}
    notifier.deliver_with = lambda: (_ for _ in ()).throw(dn.Unavailable("no_service"))
    assert api.show_native_notification("t", "", True, "n6-k")["status"] == "unavailable"
    notifier.answer = RuntimeError("probe failed")
    assert api.request_native_notifications()["reason"] == "RuntimeError: probe failed"


# --------------------------------------------------------------------------- the click route ---

class Window:
    def __init__(self, fail=False):
        self.calls, self.fail = [], fail

    def show(self):
        self.calls.append("show")

    def evaluate_js(self, script):
        if self.fail:
            raise RuntimeError("page gone")
        self.calls.append(("js", script))


def background_with(window, monkeypatch):
    monkeypatch.setattr(lb, "indicator_class", lambda: None)
    background = lb.Background(lambda: None, lambda: 8765, threading.Event())
    background.window = window
    background._can_show = True
    return background


def test_a_click_opens_the_window_and_hands_the_page_its_token(monkeypatch):
    window = Window()
    background_with(window, monkeypatch).open_notification("n3-abc")
    assert window.calls[0] == "show", "the window opens first, as it was left"
    kind, script = window.calls[1]
    assert script == f"window.ouroNotifications && window.ouroNotifications.activate({json.dumps('n3-abc')})"


def test_a_click_without_a_token_or_a_page_only_opens(monkeypatch, caplog):
    window = Window()
    background_with(window, monkeypatch).open_notification("")
    assert window.calls == ["show"]
    broken = Window(fail=True)
    background_with(broken, monkeypatch).open_notification("n3-abc")
    assert broken.calls == ["show"]
    assert "could not be told" in caplog.text


def test_the_platform_click_never_waits_on_the_callback_thread():
    clicked, gate = [], threading.Event()

    def on_click(token):
        gate.wait(5)
        clicked.append((token, threading.current_thread().name))

    Recorder(on_click=on_click)._clicked("n9-z")  # returns although on_click blocks
    gate.set()
    wait_for(lambda: clicked)
    assert clicked == [("n9-z", "ouroboros-notification-click")]


@pytest.mark.parametrize("indicator", [None, launcher_tray.WindowsTray])
def test_the_background_owns_the_platform_adapter(monkeypatch, indicator):
    seen = {}
    monkeypatch.setattr(lb, "indicator_class", lambda: indicator)
    monkeypatch.setattr(lb, "native_notifier",
                        lambda on_click, shown_on: seen.update(click=on_click, shown_on=shown_on) or "adapter")
    background = lb.Background(lambda: None, lambda: 8765, threading.Event())
    assert background.notifications == "adapter"
    assert seen["click"] == background.open_notification, "only the click route: no sound of its own"
    assert seen["shown_on"] is background.indicator, "Windows' balloons ride the launcher's one icon"
    assert not hasattr(lb, "_notifier"), "nothing of a notifier outlives the process: exit and Panic stop the icon"


@pytest.mark.parametrize(("platform", "kind"), [("darwin", dn.MacNotifier), ("win32", dn.WindowsNotifier),
                                                ("linux", dn.FreedesktopNotifier), ("freebsd14", type(None))])
def test_one_adapter_per_platform(monkeypatch, platform, kind):
    monkeypatch.setattr(dn.sys, "platform", platform)
    assert isinstance(dn.native_notifier(lambda token: None), kind)


# ------------------------------------------------------------------------------------- macOS ---

class Settings:
    def __init__(self, status, sound):
        self.status, self.sound = status, sound

    def authorizationStatus(self):
        return self.status

    def soundSetting(self):
        return self.sound


class Center:
    def __init__(self, status=0, grant=True, error=None, sound=2):
        self.status, self.grant, self.error, self.asked, self.added = status, grant, error, [], []
        self.sound, self.reads = sound, 0

    def getNotificationSettingsWithCompletionHandler_(self, handler):
        self.reads += 1
        handler(Settings(self.status, self.sound))

    def requestAuthorizationWithOptions_completionHandler_(self, options, handler):
        self.asked.append(options)
        self.status = 2 if self.grant else 1
        handler(self.grant, None)

    def addNotificationRequest_withCompletionHandler_(self, request, handler):
        self.added.append(request)
        error = self.error
        if error is None and self.status != 2:
            error = SimpleNamespace(domain=lambda: "UNErrorDomain", code=lambda: 1,
                                    localizedDescription=lambda: "Notifications are not allowed")
        handler(error)


class Content:
    def __init__(self):
        self.fields = {}

    @classmethod
    def alloc(cls):
        return cls()

    def init(self):
        return self

    def __getattr__(self, name):
        if name.startswith("set") and name.endswith("_"):
            return lambda value: self.fields.__setitem__(name[3:-1].lower(), value)
        raise AttributeError(name)


MAC_CLASSES = {
    "UNMutableNotificationContent": Content,
    "UNNotificationSound": SimpleNamespace(defaultSound=lambda: "default-sound"),
    "UNNotificationRequest": SimpleNamespace(
        requestWithIdentifier_content_trigger_=lambda identifier, content, trigger: (identifier, content, trigger)),
}


def mac(monkeypatch, center):
    built = []
    monkeypatch.setattr(dn, "_mac_center", lambda clicked: built.append(clicked) or (center, MAC_CLASSES))
    return dn.MacNotifier(lambda token: None), built


def test_macos_reads_the_permission_without_asking(monkeypatch):
    for status, expected in ((0, "not_determined"), (1, "denied"), (2, "authorized"), (3, "authorized")):
        center = Center(status=status)
        notifier, _ = mac(monkeypatch, center)
        assert notifier.status()["status"] == expected
        assert center.asked == []


@pytest.mark.parametrize("value,expected", [(2, "enabled"), (1, "disabled"), (0, "unknown"), (99, "unknown")])
def test_macos_sound_setting_is_independent_of_authorization(monkeypatch, value, expected):
    center = Center(status=2, sound=value)
    notifier, _ = mac(monkeypatch, center)
    info = bridge(notifier).shell_info()["native_notifications"]
    assert info["status"] == "authorized" and info["sound_setting"] == expected
    center.sound = 1 if value == 2 else 2
    assert notifier.status()["sound_setting"] == ("disabled" if value == 2 else "enabled")
    assert center.reads == 2 and center.asked == [], "every read returns current OS facts"


def test_macos_unreadable_sound_is_unknown(monkeypatch):
    center = Center(status=2)
    monkeypatch.setattr(Settings, "soundSetting", lambda self: (_ for _ in ()).throw(AttributeError("missing")))
    notifier, _ = mac(monkeypatch, center)
    assert notifier.status()["sound_setting"] == "unknown"


def test_macos_asks_once_for_alert_and_sound_never_a_badge(monkeypatch):
    center = Center(status=0)
    notifier, _ = mac(monkeypatch, center)
    assert notifier.ask()["status"] == "authorized"
    assert center.asked == [2 | 4]
    assert notifier.ask()["status"] == "authorized" and center.asked == [2 | 4], "decided: never asked again"


def test_macos_delivers_with_the_system_sound_and_the_pages_token(monkeypatch):
    center = Center(status=2)
    notifier, _ = mac(monkeypatch, center)
    answer = notifier.notify("Task finished", "Report ready", True, "n1-abc")
    assert answer == {"ok": True, "status": "submitted", "platform": "macos", "sound": "os"}
    identifier, content, trigger = center.added[0]
    assert (identifier, trigger) == ("n1-abc", None), "immediate, identified by the page's token"
    assert content.fields == {"title": "Task finished", "body": "Report ready", "sound": "default-sound"}

    silent = notifier.notify("Task finished", "", False, "n2-abc")
    assert silent["sound"] == "off"
    assert center.added[1][1].fields == {"title": "Task finished"}, "no body, no sound: private and silent"


def test_macos_denial_and_errors_are_typed_fallbacks(monkeypatch):
    denied = Center(status=1)
    notifier, _ = mac(monkeypatch, denied)
    assert notifier.notify("t", "", True, "n1-a")["status"] == "denied"
    assert len(denied.added) == 1 and denied.reads == 1, "the OS enforces permission; only its refusal is read after"

    error = SimpleNamespace(domain=lambda: "OtherDomain", code=lambda: 1,
                            localizedDescription=lambda: "Notifications are not allowed")
    failing = Center(status=2, error=error)
    notifier, _ = mac(monkeypatch, failing)
    answer = notifier.notify("t", "", True, "n1-a")
    assert answer["status"] == "failed" and answer["reason"].startswith("OtherDomain 1:")
    assert failing.reads == 0, "a failure is not a permission refusal: nothing to read"


@pytest.mark.parametrize("domain,code,expected", [("UNErrorDomain", 1, "denied"),
                                                ("UNErrorDomain", 2, "failed"), ("Other", 1, "failed")])
def test_macos_denial_uses_domain_and_code_not_translated_text(monkeypatch, domain, code, expected):
    error = SimpleNamespace(domain=lambda: domain, code=lambda: code,
                            localizedDescription=lambda: "Разрешения изменены")
    notifier, _ = mac(monkeypatch, Center(status=2, error=error))
    assert notifier.notify("t", "", True, "n1-a")["status"] == expected


def test_macos_an_alert_never_asks_for_permission(monkeypatch):
    """Only a gesture asks. Even a caller bypassing the page's precheck cannot ask through delivery, and the
    system's "not allowed" for an unasked question stays unasked rather than becoming the owner's denial."""
    unasked = Center(status=0)
    notifier, _ = mac(monkeypatch, unasked)
    assert notifier.notify("t", "", True, "n1-a") == {"ok": False, "status": "not_determined", "platform": "macos",
                                                      "reason": ""}
    assert unasked.asked == [] and len(unasked.added) == 1 and unasked.reads == 1
    assert notifier.ask()["status"] == "authorized" and unasked.asked == [2 | 4], "the gesture asks, once"
    assert notifier.notify("t", "", True, "n2-a")["status"] == "submitted"


def test_macos_an_authorization_error_is_typed_never_a_denial(monkeypatch):
    """The system answering its question with an NSError (no prompt shown: an app identity it refuses,
    say) is not the owner's denial: the error is the typed reason, here and on the alert that follows."""
    class Refusing(Center):
        def requestAuthorizationWithOptions_completionHandler_(self, options, handler):
            self.asked.append(options)  # the system's settings stay not_determined
            handler(False, SimpleNamespace(domain=lambda: "UNErrorDomain", code=lambda: 1,
                                           localizedDescription=lambda: "Notifications are not allowed"))

    reason = "authorization_error: UNErrorDomain 1: Notifications are not allowed"
    center = Refusing(status=0)
    notifier, _ = mac(monkeypatch, center)
    assert notifier.ask() == {"available": False, "status": "unavailable", "platform": "macos", "reason": reason,
                              "sound_setting": "unknown"}
    assert notifier.status()["status"] == "not_determined", "the system's own setting is reported as it is"
    assert notifier.notify("t", "", True, "n1-a") == {"ok": False, "status": "not_determined", "platform": "macos",
                                                      "reason": reason}
    # The owner's own refusal, with no error, is still a denial.
    notifier, _ = mac(monkeypatch, Center(status=0, grant=False))
    assert notifier.ask() == {"available": False, "status": "denied", "platform": "macos", "reason": "",
                              "sound_setting": "enabled"}


@pytest.mark.parametrize("read", ["unanswered", "broken"])
def test_macos_a_refusal_whose_permission_cannot_be_read_stays_a_denial(monkeypatch, read):
    """The one read after a definite refusal can only narrow it to unasked; without an answer it stays denied."""
    class Unreadable(Center):
        def getNotificationSettingsWithCompletionHandler_(self, handler):
            self.reads += 1
            if read == "broken":
                raise RuntimeError("settings unavailable")  # "unanswered": the completion never comes

    monkeypatch.setattr(dn, "_REPLY_WAIT", 0.05)
    center = Unreadable(status=0)
    notifier, _ = mac(monkeypatch, center)
    answer = notifier.notify("t", "", True, "n1-a")
    assert answer["status"] == "denied" and answer["reason"].startswith("UNErrorDomain 1:")
    assert center.reads == 1 and center.asked == [] and len(center.added) == 1


def test_macos_an_unanswered_question_stays_not_determined(monkeypatch):
    class Ignored(Center):
        def requestAuthorizationWithOptions_completionHandler_(self, options, handler):
            self.asked.append(options)  # the owner has not answered the system's question yet

    monkeypatch.setattr(dn, "_ASK_WAIT", 0.05)
    notifier, _ = mac(monkeypatch, Ignored(status=0))
    assert notifier.ask() == {"available": True, "status": "not_determined", "platform": "macos", "reason": "",
                              "sound_setting": "unknown"}


def test_macos_an_added_request_without_an_answer_is_unknown_and_keeps_its_click(monkeypatch):
    """Timeout after submission, then late success: no fallback (it may still appear), and the click still
    carries the page's token — the request's identifier — through the delegate."""
    class Slow(Center):
        def addNotificationRequest_withCompletionHandler_(self, request, handler):
            self.added.append(request)
            self.late = handler  # the center answers after the bridge stopped waiting

    monkeypatch.setattr(dn, "_REPLY_WAIT", 0.05)
    center = Slow(status=2)
    notifier, _ = mac(monkeypatch, center)
    assert notifier.notify("t", "", True, "n1-late") == {"ok": None, "status": "unknown", "platform": "macos",
                                                         "reason": "no_completion"}
    center.late(None)  # the late success lands harmlessly
    assert center.added[0][0] == "n1-late", "the system's click hands back this token whenever it comes"


def test_macos_submission_has_no_second_permission_wait(monkeypatch):
    class Mute(Center):
        def getNotificationSettingsWithCompletionHandler_(self, handler):
            pass

    monkeypatch.setattr(dn, "_REPLY_WAIT", 0.05)
    center = Mute(status=2)
    notifier, _ = mac(monkeypatch, center)
    assert notifier.status()["sound_setting"] == "unknown"
    answer = notifier.notify("t", "", False, "n1-a")
    assert answer["status"] == "submitted" and answer["sound"] == "off"
    assert len(center.added) == 1, "delivery submits directly after the page's final preference check"


def test_macos_nil_completion_is_submission_not_permission_proof(monkeypatch):
    """Native QA P9: usernoted accepts requests with nil error even after revocation.

    The page must prepare before its last mute check. The adapter must neither
    insert a later permission wait nor relabel an accepted request as unsent.
    """
    class Revoked(Center):
        def addNotificationRequest_withCompletionHandler_(self, request, handler):
            self.added.append(request)
            handler(None)

    center = Revoked(status=1)
    notifier, _ = mac(monkeypatch, center)
    assert notifier.status()["status"] == "denied"
    assert notifier.notify("t", "", False, "n1-a")["status"] == "submitted"
    assert center.reads == 1 and center.asked == [] and len(center.added) == 1


def test_macos_outside_an_app_bundle_is_unavailable_once(monkeypatch):
    attempts = []

    def no_bundle(clicked):
        attempts.append(1)
        raise dn.Unavailable("not_an_app_bundle")

    monkeypatch.setattr(dn, "_mac_center", no_bundle)
    notifier = dn.MacNotifier(lambda token: None)
    assert notifier.status() == {"available": False, "status": "unavailable", "platform": "macos",
                                 "reason": "not_an_app_bundle", "sound_setting": "unknown"}
    assert notifier.notify("t", "", True, "n1-a")["status"] == "unavailable"
    assert attempts == [1], "no retry storm against a missing identity"


@pytest.mark.skipif(sys.platform != "darwin", reason="PyObjC and the UserNotifications framework are macOS-only")
def test_macos_block_metadata_through_real_pyobjc():
    """The registered metadata equals pyobjc-framework-UserNotifications 12.2.1's (compared when written):
    blocks cross the bridge with the right types, and the delegate shows a frontmost app's notification."""
    objc = pytest.importorskip("objc")
    objc.loadBundle("UserNotifications", {}, bundle_path=dn._MAC_FRAMEWORK)
    delegate_class = dn._mac_delegate_class()
    center = objc.lookUpClass("UNUserNotificationCenter")
    callable_of = lambda method: method.__metadata__()["arguments"][-1]["callable"]["arguments"]  # noqa: E731
    assert [arg["type"] for arg in callable_of(center.requestAuthorizationWithOptions_completionHandler_)] == \
        [b"^v", b"Z", b"@"]
    assert [arg["type"] for arg in callable_of(center.addNotificationRequest_withCompletionHandler_)] == [b"^v", b"@"]
    assert delegate_class.userNotificationCenter_willPresentNotification_withCompletionHandler_.signature == \
        b"v@:@@@?"

    content = objc.lookUpClass("UNMutableNotificationContent").alloc().init()
    content.setTitle_("Task finished")
    request = objc.lookUpClass("UNNotificationRequest").requestWithIdentifier_content_trigger_("n1-abc", content, None)
    assert (request.identifier(), request.content().title(), request.trigger()) == ("n1-abc", "Task finished", None)

    delegate, seen = delegate_class.alloc().init(), []
    delegate.userNotificationCenter_willPresentNotification_withCompletionHandler_(None, None, seen.append)
    assert seen == [2 | 4 | 8 | 16], "Sound | Alert | List | Banner: shown in front too (2A)"

    class Response:
        def __init__(self, action):
            self.action = action

        def actionIdentifier(self):
            return self.action

        def notification(self):
            return SimpleNamespace(request=lambda: SimpleNamespace(identifier=lambda: "n7-tok"))

    clicked, finished = [], []
    saved = dn._mac_shared.get("clicked")
    dn._mac_shared["clicked"] = clicked.append
    try:
        for action in (dn._MAC_DEFAULT_ACTION, "com.apple.UNNotificationDismissActionIdentifier"):
            delegate.userNotificationCenter_didReceiveNotificationResponse_withCompletionHandler_(
                None, Response(action), lambda: finished.append(action))
    finally:
        dn._mac_shared["clicked"] = saved
    assert clicked == ["n7-tok"], "only the notification's own click opens its source"
    assert len(finished) == 2, "the system's completion handler is always called"


# ----------------------------------------------------------------------------------- Windows ---

class NetEvent:
    """A .NET event: ``+=`` subscribes; WinForms raises it on its icon's STA thread."""

    def __init__(self):
        self.handlers = []

    def __iadd__(self, handler):
        self.handlers.append(handler)
        return self

    def fire(self, sender):
        for handler in list(self.handlers):
            handler(sender, None)


class StaForms:
    """Stand-in pythonnet and WinForms modules around the production pump.

    ``Thread.Start`` runs the pump on a Python thread whose ``Application.Run`` is a message loop: its
    timers tick and the events posted to it (a click, a close) run there, as on Windows. Only the
    shell's own balloon call, ``launcher_tray._shell_balloon``, is replaced: ``shell_answer`` decides
    what the shell says, and every call is recorded as ``(hwnd, uid, title, body, silent)``."""

    def __init__(self, monkeypatch):
        self.stop, self.paused = threading.Event(), threading.Event()
        self.threads, self.icons, self.timers, self.balloons, self.winforms_balloons = [], [], [], [], []
        self.shell_answer = lambda *call: True
        self.run_error = None
        self.private_fields = True
        self.ticks = 0
        self._inbox, self._timers, self._exited = {}, {}, set()
        monkeypatch.setattr(launcher_tray, "_shell_balloon", self._shell)
        monkeypatch.setattr(dn.sys, "platform", "win32")
        for name, module in self._modules().items():
            monkeypatch.setitem(sys.modules, name, module)

    def _shell(self, *call):
        self.balloons.append(call)
        return self.shell_answer(*call)

    def post(self, sta, action):
        self._inbox.setdefault(sta, queue.SimpleQueue()).put(action)

    def click(self, icon):
        self.post(icon.sta, lambda: icon.BalloonTipClicked.fire(icon))

    def close(self, icon):
        self.post(icon.sta, lambda: icon.BalloonTipClosed.fire(icon))

    def settle(self):
        """Two more timer ticks: whatever was queued or retired before now has been handled."""
        target = self.ticks + 2
        wait_for(lambda: self.ticks >= target)

    def shutdown(self):
        self.stop.set()
        for thread in self.threads:
            thread.join(5)

    def _modules(self):
        forms = self

        class NotifyIcon:
            def __init__(self):
                self.Visible, self.Icon, self.Text, self.ContextMenuStrip, self.disposed = False, None, "", None, False
                self.BalloonTipClicked, self.BalloonTipClosed, self.MouseClick = NetEvent(), NetEvent(), NetEvent()
                self.sta, self.id = threading.get_ident(), len(forms.icons) + 1
                self.window = SimpleNamespace(Handle=SimpleNamespace(ToInt64=lambda: 0x5000 + self.id))
                forms.icons.append(self)

            def GetType(self):
                fields = ("window", "id") if forms.private_fields else ()
                return SimpleNamespace(GetField=lambda name, flags: SimpleNamespace(
                    GetValue=lambda icon: getattr(icon, name)) if name in fields else None)

            def ShowBalloonTip(self, timeout, title, body, kind):
                forms.winforms_balloons.append((self.id, title, body))

            def Dispose(self):
                self.disposed, self.Visible = True, False

        class Timer:
            def __init__(self):
                self.Interval, self.Tick, self.owner, self.disposed = 0, NetEvent(), threading.get_ident(), False
                forms.timers.append(self)

            def Start(self):
                forms._timers.setdefault(self.owner, []).append(self)

            def Stop(self):
                forms._timers.get(self.owner, []).remove(self)

            def Dispose(self):
                self.disposed = True

        class Application:
            @staticmethod
            def Run(context):
                if forms.run_error:
                    raise forms.run_error
                me = threading.get_ident()
                forms._exited.discard(me)  # a new STA may reuse an OS thread id
                inbox = forms._inbox.setdefault(me, queue.SimpleQueue())
                while not forms.stop.is_set() and me not in forms._exited:
                    if not forms.paused.is_set():
                        for timer in list(forms._timers.get(me, ())):
                            timer.Tick.fire(timer)
                        forms.ticks += 1
                    try:
                        inbox.get(timeout=0.005)()
                    except queue.Empty:
                        pass

            @staticmethod
            def ExitThread():
                forms._exited.add(threading.get_ident())

        class Thread:
            def __init__(self, start):
                self.start, self.IsBackground = start, False

            def SetApartmentState(self, state):
                pass

            def Start(self):
                thread = threading.Thread(target=self.start, name="sta-stand-in", daemon=True)
                forms.threads.append(thread)
                thread.start()

        class MenuItem:
            def __init__(self, text=""):
                self.Text, self.Enabled, self.Click = text, True, NetEvent()

        menu = lambda: SimpleNamespace(Items=SimpleNamespace(Add=lambda item: None))  # noqa: E731
        modules = {name: types.ModuleType(name) for name in (
            "clr", "System", "System.Drawing", "System.Threading", "System.Windows", "System.Windows.Forms",
            "System.Reflection")}
        modules["clr"].AddReference = lambda name: None
        for name in ("System", "System.Windows"):
            modules[name].__path__ = []
        modules["System.Drawing"].__dict__.update(Icon=lambda path: ("icon", path),
                                                  SystemIcons=SimpleNamespace(Application="app-icon"))
        modules["System.Threading"].__dict__.update(ApartmentState=SimpleNamespace(STA="sta"), Thread=Thread,
                                                    ThreadStart=lambda start: start)
        modules["System.Windows.Forms"].__dict__.update(
            Application=Application, ApplicationContext=object, NotifyIcon=NotifyIcon, Timer=Timer,
            ToolTipIcon=SimpleNamespace(Info="info"), ContextMenuStrip=menu, MouseButtons=SimpleNamespace(Left="left"),
            ToolStripMenuItem=MenuItem, ToolStripSeparator=MenuItem,
            CloseReason=SimpleNamespace(UserClosing="user"))
        modules["System.Reflection"].BindingFlags = SimpleNamespace(Instance=4, NonPublic=32)
        return modules


@pytest.fixture
def forms(monkeypatch):
    stand_in = StaForms(monkeypatch)
    yield stand_in
    stand_in.shutdown()


def start_icon(forms):
    """The launcher's one Windows icon around the stand-in WinForms, as ``Background.run`` brings it up;
    ``background.opened`` records what a click opened (a token, or "window")."""
    background = SimpleNamespace(shutdown=threading.Event(), window=None, opened=[], exit_launcher=lambda: None)
    background.show_window = lambda: background.opened.append("window")
    background.open_notification = lambda token: background.opened.append(token)
    tray = launcher_tray.WindowsTray(background)
    background._indicator_up = lambda: tray.start()
    assert tray.start()
    wait_for(tray.ready.is_set)
    return tray


@pytest.fixture
def icon(forms, monkeypatch):
    monkeypatch.setattr(lb, "_active", None)
    tray = start_icon(forms)
    yield tray
    tray.stop(wait=2.0)


def left_click(icon):
    for handler in icon.MouseClick.handlers:
        handler(icon, SimpleNamespace(Button="left"))


def test_setup_window_exit_reopens_the_single_tray_owner(icon, forms):
    from System.Windows.Forms import Application

    first = icon._icon
    forms.post(first.sta, Application.ExitThread)  # pywebview's setup window ends the shared loops
    wait_for(lambda: len(forms.icons) == 2 and icon.ready.is_set())
    assert first.disposed and not first.Visible
    assert sum(bool(item.Visible) for item in forms.icons) == 1
    assert icon._icon is forms.icons[-1]


def test_windows_a_balloon_counts_only_once_the_shell_accepted_it(forms, icon):
    notifier = dn.WindowsNotifier(lambda token: None, icon)
    assert notifier.notify("Task finished", "", True, "n1-a") == {
        "ok": True, "status": "submitted", "platform": "windows", "sound": "os"}
    (tray_icon,) = forms.icons
    assert forms.balloons == [(0x5000 + tray_icon.id, tray_icon.id, "Task finished", "Open Ouroboros to see it.",
                               False)], "sent to the one icon's window and id, so its click comes back to that icon"

    forms.shell_answer = lambda *call: False  # Shell_NotifyIconW said FALSE
    assert notifier.notify("Task finished", "", True, "n2-a") == {
        "ok": False, "status": "failed", "platform": "windows", "reason": "balloon_not_submitted"}
    forms.settle()
    assert forms.icons == [tray_icon] and tray_icon.Visible and not tray_icon.disposed, "a refusal keeps the one icon"


def test_windows_sound_off_asks_the_shell_for_a_silent_balloon(forms, icon):
    answer = dn.WindowsNotifier(lambda token: None, icon).notify("Task finished", "Report ready", False, "n1-a")
    assert answer["sound"] == "off"
    assert forms.balloons[-1][2:] == ("Task finished", "Report ready", True), "NIIF_NOSOUND"


def test_windows_missing_icon_identity_never_turns_a_silent_request_into_sound(forms, icon):
    forms.private_fields = False
    notifier = dn.WindowsNotifier(lambda token: None, icon)
    assert notifier.notify("Task finished", "", False, "n1-a")["status"] == "failed"
    assert forms.balloons == [] and forms.winforms_balloons == []
    # WinForms' ShowBalloonTip returns nothing and discards the shell's answer: handed over, unconfirmed.
    assert notifier.notify("Task finished", "", True, "n2-a") == {
        "ok": None, "status": "unknown", "platform": "windows", "reason": "balloon_unconfirmed"}
    assert [title for _id, title, _body in forms.winforms_balloons] == ["Task finished"]
    forms.click(forms.icons[0])
    wait_for(lambda: icon.background.opened)
    assert icon.background.opened == ["n2-a"], "it may still show, so it is what a click opens"


def test_windows_a_balloon_click_opens_the_newest_source_never_a_refused_one(forms, icon):
    """Owner decision 1A: one icon, so a click (an older entry in Windows' list included) opens the newest
    balloon the shell took; a refused balloon never takes the click from it."""
    notifier = dn.WindowsNotifier(lambda token: None, icon)
    assert notifier.notify("A", "", True, "n1-a")["status"] == "submitted"
    assert notifier.notify("B", "", True, "n2-b")["status"] == "submitted"
    forms.shell_answer = lambda *call: False
    assert notifier.notify("C", "", True, "n3-c")["status"] == "failed"
    (tray_icon,) = forms.icons
    forms.click(tray_icon)
    wait_for(lambda: icon.background.opened)
    assert icon.background.opened == ["n2-b"]
    forms.post(tray_icon.sta, lambda: left_click(tray_icon))
    wait_for(lambda: len(icon.background.opened) == 2)
    assert icon.background.opened[-1] == "window", "a left click on the icon opens the window, not a source"


def test_windows_notifications_never_add_an_icon(forms, icon):
    notifier = dn.WindowsNotifier(lambda token: None, icon)
    for number in range(5):
        assert notifier.notify("t", "", True, f"n{number}-a")["status"] == "submitted"
        forms.close(forms.icons[0])  # timed out or dismissed: nothing to retire, nothing added
    forms.settle()
    assert len(forms.icons) == 1 and forms.icons[0].Visible and not forms.icons[0].disposed
    assert len(forms.balloons) == 5, "nothing is shown again"


def test_windows_an_unanswered_balloon_is_unknown_and_its_late_success_takes_the_click(forms, icon, monkeypatch):
    """Timeout, then late success: no fallback (it may still appear), and it becomes the click target."""
    monkeypatch.setattr(launcher_tray, "SUBMIT_WAIT", 0.2)
    release = threading.Event()
    forms.shell_answer = lambda *call: release.wait(5)  # the shell call outlasts the caller's wait
    assert dn.WindowsNotifier(lambda token: None, icon).notify("Task finished", "", True, "n1-late") == {
        "ok": None, "status": "unknown", "platform": "windows", "reason": "balloon_unconfirmed"}
    release.set()
    forms.settle()
    forms.click(forms.icons[0])
    wait_for(lambda: icon.background.opened)
    assert icon.background.opened == ["n1-late"]


def test_windows_a_balloon_the_pump_never_took_is_refused_and_never_the_click(forms, icon, monkeypatch):
    monkeypatch.setattr(launcher_tray, "SUBMIT_WAIT", 0.2)
    notifier = dn.WindowsNotifier(lambda token: None, icon)
    assert notifier.notify("first", "", True, "n0-a")["status"] == "submitted"
    forms.paused.set()  # the STA thread is busy elsewhere: no tick reaches the queue
    assert notifier.notify("Task finished", "", True, "n1-a") == {
        "ok": False, "status": "failed", "platform": "windows", "reason": "balloon_not_submitted"}
    forms.paused.clear()
    forms.settle()
    assert [call[2] for call in forms.balloons] == ["first"], "given up on, so the page's fallback is the only alert"
    forms.click(forms.icons[0])
    wait_for(lambda: icon.background.opened)
    assert icon.background.opened == ["n0-a"]


def test_windows_an_unreadable_icon_asset_keeps_the_icon(forms, monkeypatch, tmp_path):
    """A broken ``icon.ico`` falls back to the system's application icon."""
    from ouroboros import platform_layer

    (tmp_path / "assets").mkdir()
    (tmp_path / "assets" / "icon.ico").write_bytes(b"not an icon")
    monkeypatch.setattr(platform_layer, "bundled_resource_bases", lambda: [tmp_path])
    monkeypatch.setattr(lb, "_active", None)

    def unreadable(path):
        raise OSError(f"bad icon: {path}")

    monkeypatch.setattr(sys.modules["System.Drawing"], "Icon", unreadable)
    tray = start_icon(forms)
    try:
        assert dn.WindowsNotifier(lambda token: None, tray).notify("t", "", True, "n1-a")["status"] == "submitted"
        assert forms.icons[0].Icon == "app-icon"
    finally:
        tray.stop(wait=2.0)


def test_windows_without_a_live_icon_a_notification_is_unavailable_and_starts_nothing(forms):
    unavailable = {"ok": False, "status": "unavailable", "platform": "windows", "reason": "notification_area_unavailable"}
    assert dn.WindowsNotifier(lambda token: None).notify("t", "", True, "n1-a") == unavailable
    background = SimpleNamespace(shutdown=threading.Event(), window=None)
    assert dn.WindowsNotifier(lambda token: None, launcher_tray.WindowsTray(background)).notify(
        "t", "", True, "n2-b") == unavailable
    assert forms.threads == [] and forms.icons == [], "a notification never starts the icon"


def test_windows_a_dead_pump_is_unavailable(forms, monkeypatch):
    monkeypatch.setattr(lb, "_active", None)
    forms.run_error = RuntimeError("no message loop")
    background = SimpleNamespace(shutdown=threading.Event(), window=None, exit_launcher=lambda: None)
    tray = launcher_tray.WindowsTray(background)
    tray.start()
    wait_for(lambda: tray._idle.is_set())  # the pump ended without showing anything
    assert dn.WindowsNotifier(lambda token: None, tray).notify("t", "", True, "n1-a") == {
        "ok": False, "status": "unavailable", "platform": "windows", "reason": "notification_area_unavailable"}


def test_windows_exit_removes_the_icon_its_timer_and_its_pump_and_never_brings_it_back(forms, icon):
    """The launcher's exit (``launcher_background.stop_tray_before_exit``): the one icon may not outlive the
    process in the notification area, the timer and the STA loop end with it, and no notification
    afterwards starts another icon."""
    notifier = dn.WindowsNotifier(lambda token: None, icon)
    assert notifier.notify("t", "", True, "n1-a")["status"] == "submitted"
    (timer,) = forms.timers
    released = []
    lb.stop_tray_before_exit(lambda: released.append(True))  # waits for the icon's removal, bounded
    assert released == [True] and all(item.disposed and not item.Visible for item in forms.icons)
    for thread in forms.threads:
        thread.join(2)
    assert not any(thread.is_alive() for thread in forms.threads), "the STA loop ended"
    assert timer.disposed and not any(forms._timers.values()), "then the timer is stopped and disposed"
    assert notifier.notify("t", "", True, "n2-b") == {
        "ok": False, "status": "unavailable", "platform": "windows", "reason": "notification_area_unavailable"}
    assert len(forms.icons) == 1, "no icon after the exit began"


@pytest.mark.parametrize("refused,acquired", [(False, True), (True, True), (True, False)])
def test_launcher_handoff_removes_the_live_icon_before_unlock_and_restores_only_when_locked(
        forms, monkeypatch, tmp_path, refused, acquired):
    """Real launcher → transport → Background → WindowsTray, with WinForms/OS effects stood in.
    A spawn sees zero old icons. Refusal restores one usable icon only after retaking the lock."""
    import launcher
    from ouroboros import platform_layer
    from tests.test_launcher_source_handoff import context, Exited

    monkeypatch.setattr("ouroboros.launcher_bootstrap.source_launcher_importable", lambda *a, **kw: True)
    events = []
    monkeypatch.setattr(lb, "_active", None)
    monkeypatch.setattr(lb.Background, "_watch", lambda self: None)
    background = lb.Background(lambda: events.append("quit"), lambda: 1, threading.Event())
    background.window = SimpleNamespace(show=lambda: events.append("show"))
    background._window_shown()
    assert background._indicator_up()
    old_icon = forms.icons[0]
    background.indicator.begin_hidden()

    def release():
        events.append("release")
        assert old_icon.disposed and not old_icon.Visible, "old icon removed while the PID lock is held"

    def spawn(*args, **kwargs):
        events.append("spawn")
        assert not any(icon.Visible for icon in forms.icons), "no overlapping icons at successor spawn"
        if refused:
            raise OSError("definite spawn refusal")

    def leave(code):
        events.append(("exit", code))
        raise Exited(code)

    monkeypatch.setattr(launcher, "_bootstrap_context", lambda: context(tmp_path))
    monkeypatch.setattr(launcher, "release_pid_lock", release)
    monkeypatch.setattr(launcher, "acquire_pid_lock", lambda: events.append("acquire") or acquired)
    monkeypatch.setattr(platform_layer, "IS_WINDOWS", True)
    monkeypatch.setattr(platform_layer, "inherited_console_kwargs", lambda: {})
    monkeypatch.setattr(launcher.subprocess, "Popen", spawn)
    monkeypatch.setattr(launcher.os, "_exit", leave)
    if refused and acquired:
        launcher._exec_source_launcher()
        assert events == ["release", "spawn", "acquire"]
        assert len(forms.icons) == 2 and sum(icon.Visible for icon in forms.icons) == 1
        assert background.notifications.notify("restored", "", False, "n1-restored")["status"] == "submitted"
        background.show_window()
        assert events[-1] == "show" and not background.indicator.hidden
    else:
        with pytest.raises(Exited) as exited:
            launcher._exec_source_launcher()
        assert exited.value.code == (1 if refused else 0)
        assert len(forms.icons) == 1 and not old_icon.Visible, "an unlocked old shell never restores an icon"
    background.indicator.stop(wait=1)


def test_windows_a_balloon_still_queued_when_the_icon_goes_is_refused_at_once(forms, icon, monkeypatch):
    monkeypatch.setattr(launcher_tray, "SUBMIT_WAIT", 30.0)  # without the refusal its caller would wait this long
    forms.paused.set()  # the STA tick is busy: the request stays queued
    answers = []
    caller = threading.Thread(target=lambda: answers.append(icon.balloon("t", "b", True, "n1-a")), daemon=True)
    caller.start()
    wait_for(lambda: not icon._balloons.empty())
    started = time.monotonic()
    icon.stop()
    forms.paused.clear()
    caller.join(5)
    assert answers == [False] and time.monotonic() - started < 5, "never shown now, so its caller falls back at once"
    assert forms.balloons == []


def test_windows_without_winforms_is_unavailable(monkeypatch):
    monkeypatch.setattr(dn.sys, "platform", "win32")
    monkeypatch.setitem(sys.modules, "clr", None)  # import clr raises ImportError
    notifier = dn.WindowsNotifier(lambda token: None)
    assert notifier.status()["status"] == "unavailable"
    assert notifier.status()["reason"].startswith("no_winforms")


def test_the_tray_banner_reports_the_shells_answer_and_honours_sound_off(forms, icon):
    """The hidden window's attention banner rides the same icon: the caller learns the shell's answer (a
    refused banner leaves the sound to the caller), Sound off is silent, and as the newest balloon its click
    opens the window as it was left (no source to navigate to)."""
    assert icon.notify("Task finished", "Something needs your attention.", False) is True
    assert forms.balloons[-1][2:] == ("Task finished", "Something needs your attention.", True)
    forms.shell_answer = lambda *call: False
    assert icon.notify("Task finished", "Something needs your attention.", True) is False
    forms.click(forms.icons[0])
    wait_for(lambda: icon.background.opened)
    assert icon.background.opened == [""]


def test_balloon_text_fits_the_shells_fields_in_utf16_units():
    assert launcher_tray._fit("a" * 70, 63) == "a" * 63
    cut = launcher_tray._fit("\U0001F600" * 40, 63)
    assert len(cut.encode("utf-16-le")) // 2 <= 63, "an emoji is two WCHARs"


@pytest.mark.skipif(sys.platform != "win32", reason="WCHAR is two bytes only on Windows")
def test_the_notify_icon_data_matches_shellapi():
    assert ctypes.sizeof(launcher_tray.notify_icon_data()) == (976 if ctypes.sizeof(ctypes.c_void_p) == 8 else 956)


# ------------------------------------------------------------------------------------- Linux ---

class Variant:
    def __init__(self, kind, value):
        self.kind, self.value = kind, value

    def __eq__(self, other):
        return isinstance(other, Variant) and (self.kind, self.value) == (other.kind, other.value)

    def __repr__(self):
        return f"Variant({self.kind!r}, {self.value!r})"

    def unpack(self):
        return self.value


class GError(Exception):
    def __init__(self, domain, code):
        super().__init__(f"{domain}:{code}")
        self.domain, self.code = domain, code

    def matches(self, domain, code):
        return (self.domain, self.code) == (domain, code)


IO_QUARK, DBUS_QUARK = "g-io-error-quark", "g-dbus-error-quark"
TIMED_OUT, NO_REPLY, SERVICE_UNKNOWN = 24, 4, 2


class Bus:
    def __init__(self, caps):
        self.caps, self.sent, self.subscribed, self.error, self.caps_error = caps, [], {}, None, None

    def call_sync(self, name, path, interface, method, parameters, reply_type, flags, timeout, cancellable):
        assert (name, path, interface) == (dn._FD_NAME, dn._FD_PATH, dn._FD_NAME)
        if method == "GetCapabilities":
            if self.caps_error:
                raise self.caps_error
            return Variant("(as)", (list(self.caps),))
        self.sent.append(parameters.value)
        if self.error:
            raise self.error
        return Variant("(u)", (40 + len(self.sent),))

    def signal_subscribe(self, sender, interface, member, path, arg0, flags, callback):
        self.subscribed[member] = callback


@pytest.fixture
def freedesktop(monkeypatch):
    def install(caps, bus_error=None):
        bus = Bus(caps)
        gio = SimpleNamespace(BusType=SimpleNamespace(SESSION="session"), DBusCallFlags=SimpleNamespace(NONE=0),
                              DBusSignalFlags=SimpleNamespace(NONE=0), io_error_quark=lambda: IO_QUARK,
                              IOErrorEnum=SimpleNamespace(TIMED_OUT=TIMED_OUT),
                              dbus_error_quark=lambda: DBUS_QUARK, DBusError=SimpleNamespace(NO_REPLY=NO_REPLY))

        def bus_get_sync(kind, cancellable):
            if bus_error:
                raise bus_error
            return bus

        gio.bus_get_sync = bus_get_sync
        glib = SimpleNamespace(Variant=Variant, VariantType=SimpleNamespace(new=lambda signature: signature),
                               markup_escape_text=lambda value: value.replace("&", "&amp;").replace("<", "&lt;"))
        repository = types.ModuleType("gi.repository")
        repository.Gio, repository.GLib = gio, glib
        gi = types.ModuleType("gi")
        gi.repository = repository
        monkeypatch.setitem(sys.modules, "gi", gi)
        monkeypatch.setitem(sys.modules, "gi.repository", repository)
        return bus

    return install


@pytest.fixture
def no_sound_of_our_own(monkeypatch):
    """1A: the notification server owns the sound; the launcher starts no player (canberra-gtk-play) and
    asks for no attention sound beside it."""
    from ouroboros import platform_layer

    played = []
    record = lambda *args, **kwargs: played.append(args) or {}  # noqa: E731
    monkeypatch.setattr(platform_layer, "request_native_attention", record)
    monkeypatch.setattr(lb, "request_native_attention", record)
    monkeypatch.setattr(subprocess, "Popen", record)
    yield played
    assert played == []


def test_linux_uses_the_servers_sound_and_its_default_action(freedesktop, no_sound_of_our_own):
    bus = freedesktop(["actions", "body", "body-markup", "sound"])
    clicked = []
    notifier = dn.FreedesktopNotifier(clicked.append)
    assert notifier.status() == {"available": True, "status": "authorized", "platform": "linux", "reason": ""}
    answer = notifier.notify("Task finished", "a < b & c", True, "n1-a")
    assert answer == {"ok": True, "status": "submitted", "platform": "linux", "sound": "os"}
    app, replaces, icon, title, body, actions, hints, expire = bus.sent[0]
    assert (app, replaces, title, body, actions, expire) == (
        "Ouroboros", 0, "Task finished", "a &lt; b &amp; c", ["default", "Open"], -1)
    assert hints["sound-name"] == Variant("s", "message-new-instant")

    notifier._clicked = clicked.append  # synchronous here; the real one starts a thread
    bus.subscribed["ActionInvoked"](None, None, None, None, None, Variant("(us)", (41, "default")))
    bus.subscribed["ActionInvoked"](None, None, None, None, None, Variant("(us)", (41, "default")))
    assert clicked == ["n1-a"], "one click, one source"


def test_linux_without_server_sound_sends_it_silently_and_plays_nothing_else(freedesktop, no_sound_of_our_own):
    assert list(inspect.signature(dn.FreedesktopNotifier).parameters) == ["on_click"], "no sound helper to call"
    bus = freedesktop(["body"])
    notifier = dn.FreedesktopNotifier(lambda token: None)
    assert notifier.status()["limits"] == ["no_sound", "no_click"], "Settings names what this server lacks"
    assert notifier.ask()["limits"] == ["no_sound", "no_click"]
    answer = notifier.notify("Task finished", "x<y", True, "n2-a")
    assert answer["sound"] == "none", "this server plays no sound, and none is played beside it"
    _app, _r, _i, _t, body, actions, hints, _e = bus.sent[0]
    assert (body, actions) == ("x<y", []), "no markup and no actions where the server offers none"
    assert "sound-name" not in hints and "suppress-sound" not in hints

    silent = notifier.notify("Task finished", "", False, "n3-a")
    assert silent["sound"] == "off"
    assert bus.sent[1][6]["suppress-sound"] == Variant("b", True)


@pytest.mark.parametrize("error", [GError(IO_QUARK, TIMED_OUT), GError(DBUS_QUARK, NO_REPLY)])
def test_linux_a_notify_without_an_answer_is_unknown(freedesktop, error):
    bus = freedesktop(["actions"])
    bus.error = error
    assert dn.FreedesktopNotifier(lambda token: None).notify("t", "", True, "n1-a") == {
        "ok": None, "status": "unknown", "platform": "linux", "reason": "no_reply"}


def test_linux_a_server_refusal_is_a_failure(freedesktop):
    bus = freedesktop(["actions"])
    bus.error = GError(DBUS_QUARK, SERVICE_UNKNOWN)
    answer = dn.FreedesktopNotifier(lambda token: None).notify("t", "", True, "n1-a")
    assert answer["status"] == "failed" and answer["ok"] is False


def test_linux_closed_notifications_are_forgotten(freedesktop):
    bus = freedesktop(["actions"])
    clicked = []
    notifier = dn.FreedesktopNotifier(clicked.append)
    notifier._clicked = clicked.append
    notifier.notify("t", "", False, "n1-a")
    bus.subscribed["NotificationClosed"](None, None, None, None, None, Variant("(uu)", (41, 2)))
    bus.subscribed["ActionInvoked"](None, None, None, None, None, Variant("(us)", (41, "default")))
    assert clicked == []


def test_linux_a_bus_that_did_not_answer_yet_is_asked_again(freedesktop):
    """A slow session bus at sign-in is not remembered: the next status or alert asks again (no timer)."""
    bus = freedesktop(["actions", "sound"])
    bus.caps_error = GError(IO_QUARK, TIMED_OUT)
    notifier = dn.FreedesktopNotifier(lambda token: None)
    assert notifier.status() == {"available": False, "status": "unavailable", "platform": "linux",
                                 "reason": "service_did_not_answer"}
    assert notifier.notify("t", "", True, "n1-a")["reason"] == "service_did_not_answer"
    assert bus.sent == [], "nothing was handed over, so the page's fallback is the only alert"
    bus.caps_error = None
    assert notifier.status()["status"] == "authorized"
    assert notifier.notify("t", "", True, "n2-a")["status"] == "submitted"


def test_linux_without_a_notification_server_is_unavailable(freedesktop):
    freedesktop([], bus_error=RuntimeError("no session bus"))
    notifier = dn.FreedesktopNotifier(lambda token: None)
    assert notifier.status() == {"available": False, "status": "unavailable", "platform": "linux",
                                 "reason": "RuntimeError: no session bus"}
    assert notifier.notify("t", "", True, "n1-a")["status"] == "unavailable"

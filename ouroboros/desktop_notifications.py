"""Native system notifications for the desktop window (DESIGN §9; owner decisions 1A, 2A, 3A).

A notification the operating system accepted (``submitted``) is the system's own: it owns its
sound, so the page plays no tone after it (1A); it is sent whether or not the window is in front
(2A); and its click opens the window and hands the page back the token the page chose, which the
page maps to the existing source activation. Submitted is the system's acceptance, not proof that
anyone saw a banner: its notification settings, Focus and Do Not Disturb still decide. A request
handed over that the system did not answer in time is ``unknown``: it may still appear, so the page
shows nothing else and keeps its click target. A refusal before the system took the request —
``not_determined``, ``denied``, ``unavailable`` or ``failed`` — is typed, and the page keeps its
browser and in-app fallbacks. Delivery never asks for permission; only the owner's own gesture
(switching notifications on, the Test button) asks the system's one question. A launcher built
before this module has none of these bridge methods; the page then falls back and says that native
delivery needs the current app (3A).

One adapter per platform, used only where the platform path is actually there:

* macOS: ``UNUserNotificationCenter``, inside an app bundle (a bare interpreter has no identity
  to authorize). The framework is loaded dynamically and the block signatures it needs are
  registered here, so no new PyObjC wrapper joins the frozen launcher.
* Windows: a notification-area balloon, which Windows 10/11 present as a system notification, on
  an icon of its own until its click (``launcher_tray.NotificationIcon``): a balloon's events name
  no balloon, so one icon per notification is what ties a click to its own source.
* Linux: the freedesktop Notifications service on the session bus through ``gi``'s Gio, when a
  notification server answers. The server owns the sound; nothing else plays one.

No scheduler, repeat, badge or stored state: one call, one notification, while the app runs.
"""
from __future__ import annotations

import itertools
import logging
import re
import sys
import threading
import time

log = logging.getLogger("launcher.notifications")

AUTHORIZED, NOT_DETERMINED, DENIED, UNAVAILABLE, FAILED = (
    "authorized", "not_determined", "denied", "unavailable", "failed")
SUBMITTED, UNKNOWN = "submitted", "unknown"
_TOKEN = re.compile(r"[A-Za-z0-9_.:-]{1,96}")
_TITLE_LIMIT, _BODY_LIMIT = 120, 400
_REPLY_WAIT = 5.0  # seconds for a notification service to answer
_ASK_WAIT = 120.0  # the owner reading the system's one permission question
_fresh = itertools.count(1)


class Unavailable(Exception):
    """The platform path is not there; the message is the typed reason."""


def capability(platform: str, status: str, reason: str = "", limits=()) -> dict:
    """``limits``: what this system lacks of what Settings describes (``no_sound``, ``no_click``)."""
    answer = {"available": status in (AUTHORIZED, NOT_DETERMINED), "status": status,
              "platform": platform, "reason": reason}
    return {**answer, "limits": list(limits)} if limits else answer


def submitted(platform: str, sound: str) -> dict:
    """The system accepted the request and decides how it appears. ``sound``: ``os`` the system's own was
    asked for; ``off`` sent silent; ``none`` this system offers no sound to ask for
    (a Linux server without one)."""
    return {"ok": True, "status": SUBMITTED, "platform": platform, "sound": sound}


def unconfirmed(platform: str, reason: str) -> dict:
    """Handed to the system, which did not answer in time: it may still appear, so never a fallback."""
    return {"ok": None, "status": UNKNOWN, "platform": platform, "reason": reason}


def refused(platform: str, status: str, reason: str = "") -> dict:
    """The system never took it: the page falls back."""
    return {"ok": False, "status": status, "platform": platform, "reason": reason}


def clean_token(raw) -> str:
    """The page's opaque click token, or a fresh one; never script text (it is echoed into the page)."""
    token = str(raw or "")
    return token if _TOKEN.fullmatch(token) else f"n{next(_fresh)}-{int(time.time())}"


def _reason(exc: BaseException) -> str:
    return str(exc) if isinstance(exc, Unavailable) else f"{type(exc).__name__}: {exc}"[:160]


def _ns_error(error) -> str:
    """An ``NSError`` as a typed reason: its domain and code, then the system's own description."""
    try:
        return f"{error.domain()} {int(error.code())}: {error.localizedDescription()}"[:160]
    except Exception:
        return f"{type(error).__name__}: {error}"[:160]


def _await(start, timeout: float):
    """Run ``start(finish)`` and wait for its completion handler; never call this on the UI thread."""
    box: list = []
    done = threading.Event()

    def finish(*values):
        box.append(values)
        done.set()

    start(finish)
    if not done.wait(timeout):
        raise TimeoutError("the notification service did not answer")
    return box[0]


class NativeNotifier:
    """The common half: input cleaning and typed answers; a platform half never raises past it."""

    platform = ""

    def __init__(self, on_click) -> None:
        self._on_click = on_click

    def status(self) -> dict:
        """The current permission, without asking the owner anything."""
        try:
            return capability(self.platform, self._status(), limits=self._limits())
        except Exception as exc:
            return capability(self.platform, UNAVAILABLE, _reason(exc))

    def ask(self) -> dict:
        """The system's one permission question where it has one (an owner gesture asks)."""
        try:
            return capability(self.platform, self._request(), limits=self._limits())
        except Exception as exc:
            return capability(self.platform, UNAVAILABLE, _reason(exc))

    def notify(self, title: str, body: str, sound: bool, token: str) -> dict:
        title = " ".join(str(title or "").split())[:_TITLE_LIMIT] or "Ouroboros"
        body = str(body or "").strip()[:_BODY_LIMIT]
        try:
            return self._deliver(title, body, bool(sound), clean_token(token))
        except Unavailable as exc:
            return refused(self.platform, UNAVAILABLE, str(exc))
        except Exception as exc:
            log.warning("Native notification failed; the page falls back.", exc_info=True)
            return refused(self.platform, FAILED, _reason(exc))

    def stop(self, wait: float = 0.0) -> None:
        """The launcher's exit or Panic: release what would outlive the process (Windows' icons)."""

    def _clicked(self, token: str) -> None:
        """A notification was clicked: off the platform's callback thread, which must not wait on the page."""
        threading.Thread(target=self._on_click, args=(str(token or ""),), name="ouroboros-notification-click",
                         daemon=True).start()

    # Platform half.
    def _status(self) -> str:
        raise NotImplementedError

    def _request(self) -> str:
        return self._status()

    def _limits(self) -> tuple:
        return ()

    def _deliver(self, title: str, body: str, sound: bool, token: str) -> dict:
        raise NotImplementedError


# --------------------------------------------------------------------------------------- macOS ---

_MAC_FRAMEWORK = "/System/Library/Frameworks/UserNotifications.framework"
_MAC_ALLOW = 2 | 4  # UNAuthorizationOptionSound | Alert: no badge (DESIGN §9)
_MAC_PRESENT = 2 | 4 | 8 | 16  # presentation Sound | Alert | List | Banner: in front too (2A)
_MAC_DEFAULT_ACTION = "com.apple.UNNotificationDefaultActionIdentifier"
_MAC_STATUS = {0: NOT_DETERMINED, 1: DENIED, 2: AUTHORIZED, 3: AUTHORIZED, 4: AUTHORIZED}
_MAC_CLASSES = ("UNUserNotificationCenter", "UNMutableNotificationContent", "UNNotificationRequest",
                "UNNotificationSound")
_mac_shared: dict = {}  # an Objective-C class is process-wide: the delegate is defined once


def _register_mac_metadata(objc) -> None:
    """The block signatures pyobjc-framework-UserNotifications would supply (PyObjC metadata format)."""
    def block(*types):
        arguments = {0: {"type": b"^v"}, **{index + 1: {"type": kind} for index, kind in enumerate(types)}}
        return {"type": b"@?", "callable": {"retval": {"type": b"v"}, "arguments": arguments}}

    register = objc.registerMetaDataForSelector
    updating = getattr(objc, "_updatingMetadata", lambda flag: None)
    updating(True)
    try:
        register(b"UNUserNotificationCenter", b"requestAuthorizationWithOptions:completionHandler:",
                 {"arguments": {3: block(b"Z", b"@")}})
        register(b"UNUserNotificationCenter", b"getNotificationSettingsWithCompletionHandler:",
                 {"arguments": {2: block(b"@")}})
        register(b"UNUserNotificationCenter", b"addNotificationRequest:withCompletionHandler:",
                 {"arguments": {3: block(b"@")}})
        for selector, handler in ((b"userNotificationCenter:willPresentNotification:withCompletionHandler:",
                                   block(b"Q")),
                                  (b"userNotificationCenter:didReceiveNotificationResponse:withCompletionHandler:",
                                   block())):
            register(b"NSObject", selector, {"required": False, "retval": {"type": b"v"},
                                             "arguments": {2: {"type": b"@"}, 3: {"type": b"@"}, 4: handler}})
    finally:
        updating(False)


def _mac_delegate_class():
    if "delegate" not in _mac_shared:
        import objc
        from Foundation import NSObject

        _register_mac_metadata(objc)

        class OuroborosNotificationDelegate(NSObject):
            def userNotificationCenter_willPresentNotification_withCompletionHandler_(self, center, note, handler):
                handler(_MAC_PRESENT)  # macOS hides a frontmost app's notifications unless asked

            def userNotificationCenter_didReceiveNotificationResponse_withCompletionHandler_(self, center, response,
                                                                                             handler):
                try:
                    if str(response.actionIdentifier()) == _MAC_DEFAULT_ACTION:
                        _mac_shared["clicked"](str(response.notification().request().identifier()))
                finally:
                    handler()

        _mac_shared["delegate"] = OuroborosNotificationDelegate
    return _mac_shared["delegate"]


def _mac_center(clicked):
    """The process's notification center with this launcher's delegate, or ``Unavailable``."""
    import objc
    from Foundation import NSBundle

    bundle = NSBundle.mainBundle()
    if not bundle.bundleIdentifier() or not str(bundle.bundlePath()).endswith(".app"):
        raise Unavailable("not_an_app_bundle")
    objc.loadBundle("UserNotifications", {}, bundle_path=_MAC_FRAMEWORK)
    delegate_class = _mac_delegate_class()
    classes = {name: objc.lookUpClass(name) for name in _MAC_CLASSES}
    center = classes["UNUserNotificationCenter"].currentNotificationCenter()
    delegate = delegate_class.alloc().init()
    _mac_shared.update(clicked=clicked, keep=delegate)  # the center holds its delegate weakly
    center.setDelegate_(delegate)
    return center, classes


class MacNotifier(NativeNotifier):
    platform = "macos"

    def __init__(self, on_click) -> None:
        super().__init__(on_click)
        self._center = None
        self._classes: dict = {}
        self._broken = ""
        self._ask_error = ""  # the system's own error from its last permission question, if it gave one
        self._lock = threading.Lock()

    def _ready(self):
        with self._lock:
            if self._center is None and not self._broken:
                try:
                    self._center, self._classes = _mac_center(self._clicked)
                except Exception as exc:
                    self._broken = _reason(exc)
                    log.info("macOS notifications unavailable: %s", self._broken)
        if self._broken:
            raise Unavailable(self._broken)
        return self._center

    def _status(self) -> str:
        center = self._ready()
        (settings,) = _await(lambda done: center.getNotificationSettingsWithCompletionHandler_(
            lambda value: done(value)), _REPLY_WAIT)
        return _MAC_STATUS.get(int(settings.authorizationStatus()), UNAVAILABLE)

    def _request(self) -> str:
        status = self._status()
        if status != NOT_DETERMINED:
            return status
        try:
            granted, error = _await(lambda done: self._center.requestAuthorizationWithOptions_completionHandler_(
                _MAC_ALLOW, lambda allowed, error: done(allowed, error)), _ASK_WAIT)
        except TimeoutError:
            return NOT_DETERMINED  # the system's question is still unanswered
        self._ask_error = "" if granted or error is None else f"authorization_error: {_ns_error(error)}"
        if self._ask_error:
            # The system answered with an error, not the owner's choice (an app identity it refuses, say).
            raise Unavailable(self._ask_error)
        return AUTHORIZED if granted else DENIED

    def _deliver(self, title, body, sound, token):
        # Only the owner's gesture asks (``ask``); an alert that finds the question unasked falls back.
        status = self._status()
        if status != AUTHORIZED:
            # Still unasked after the system's question failed: its error stays the reason.
            return refused(self.platform, status, self._ask_error if status == NOT_DETERMINED else "")
        content = self._classes["UNMutableNotificationContent"].alloc().init()
        content.setTitle_(title)
        if body:
            content.setBody_(body)
        if sound:
            content.setSound_(self._classes["UNNotificationSound"].defaultSound())
        request = self._classes["UNNotificationRequest"].requestWithIdentifier_content_trigger_(token, content, None)
        try:
            (error,) = _await(lambda done: self._center.addNotificationRequest_withCompletionHandler_(
                request, lambda value: done(value)), _REPLY_WAIT)
        except TimeoutError:
            # Added, unanswered: the center may still show it, and its click still carries the token.
            return unconfirmed(self.platform, "no_completion")
        if error is not None:
            return refused(self.platform, FAILED, str(error.localizedDescription()))
        return submitted(self.platform, "os" if sound else "off")


# ------------------------------------------------------------------------------------- Windows ---

_WINDOWS_TEXT = "Open Ouroboros to see it."  # a balloon needs text; with text hidden the title names the event


class WindowsNotifier(NativeNotifier):
    """Windows' notification settings decide how a balloon appears; Sound off asks it for silence
    (``launcher_tray.show_balloon``), which only WinForms' own call, the fallback, cannot (DESIGN §9)."""

    platform = "windows"

    def __init__(self, on_click) -> None:
        super().__init__(on_click)
        self._icons = None
        self._lock = threading.Lock()

    def _status(self) -> str:
        if sys.platform != "win32":
            raise Unavailable("not_windows")
        try:
            import clr  # noqa: F401  (pythonnet: the WinForms stack the window already runs on)
        except Exception as exc:
            raise Unavailable(f"no_winforms: {type(exc).__name__}") from exc
        return AUTHORIZED  # no per-app question; Windows' own notification settings still decide

    def _pump(self):
        with self._lock:
            if self._icons is None:
                from ouroboros.launcher_tray import NotificationIcon

                self._icons = NotificationIcon(self._clicked)
        return self._icons

    def _deliver(self, title, body, sound, token):
        self._status()
        outcome = self._pump().show(title, body or _WINDOWS_TEXT, sound, token)
        if outcome is None:
            return unconfirmed(self.platform, "balloon_unconfirmed")
        if not outcome:
            if not self._icons.alive.is_set():
                return refused(self.platform, UNAVAILABLE, "notification_area_unavailable")
            return refused(self.platform, FAILED, "balloon_not_submitted")
        return submitted(self.platform, outcome)

    def stop(self, wait: float = 0.0) -> None:
        self._pump().stop(wait)  # one never started stays unstarted: no alert after the exit began


# --------------------------------------------------------------------------------------- Linux ---

_FD_NAME = "org.freedesktop.Notifications"
_FD_PATH = "/org/freedesktop/Notifications"
_FD_SOUND = "message-new-instant"
_FD_LIMIT = 200  # clickable notifications remembered; the oldest is forgotten first


def _no_reply(Gio, exc) -> bool:
    """``Notify`` went out and no answer came back: the server may have shown it."""
    matches = getattr(exc, "matches", None)
    return callable(matches) and bool(matches(Gio.io_error_quark(), Gio.IOErrorEnum.TIMED_OUT)
                                      or matches(Gio.dbus_error_quark(), Gio.DBusError.NO_REPLY))


class FreedesktopNotifier(NativeNotifier):
    """Desktop Notifications 1.2: actions and sound are optional server capabilities, used only when offered.
    The server owns the sound (1A): one without the ``sound`` capability shows the notification silently."""

    platform = "linux"

    def __init__(self, on_click) -> None:
        super().__init__(on_click)
        self._bus = None
        self._gi = None
        self._caps: frozenset = frozenset()
        self._broken = ""
        self._ids: dict = {}
        self._lock = threading.Lock()

    def _connect(self):
        with self._lock:
            if self._bus is None and not self._broken:
                Gio = None
                try:
                    from gi.repository import Gio, GLib

                    bus = Gio.bus_get_sync(Gio.BusType.SESSION, None)
                    reply = bus.call_sync(_FD_NAME, _FD_PATH, _FD_NAME, "GetCapabilities", None,
                                          GLib.VariantType.new("(as)"), Gio.DBusCallFlags.NONE, 3000, None)
                    self._caps = frozenset(reply.unpack()[0])
                    for signal, handler in (("ActionInvoked", self._invoked), ("NotificationClosed", self._closed)):
                        bus.signal_subscribe(_FD_NAME, _FD_NAME, signal, _FD_PATH, None, Gio.DBusSignalFlags.NONE,
                                             handler)
                    self._bus, self._gi = bus, (Gio, GLib)
                except Exception as exc:
                    if Gio is not None and _no_reply(Gio, exc):
                        # A bus still slow at sign-in: not remembered, so the next call asks again.
                        raise Unavailable("service_did_not_answer") from exc
                    self._broken = _reason(exc)
                    log.info("Desktop notifications unavailable: %s", self._broken)
        if self._broken:
            raise Unavailable(self._broken)
        return self._bus

    def _invoked(self, _connection, _sender, _path, _interface, _signal, parameters) -> None:
        nid, action = parameters.unpack()
        with self._lock:
            token = self._ids.pop(int(nid), None)
        if token is not None and action == "default":
            self._clicked(token)

    def _closed(self, _connection, _sender, _path, _interface, _signal, parameters) -> None:
        with self._lock:
            self._ids.pop(int(parameters.unpack()[0]), None)

    def _status(self) -> str:
        self._connect()
        return AUTHORIZED  # the specification has no permission; the desktop's settings still decide

    def _limits(self) -> tuple:
        return tuple(limit for cap, limit in (("sound", "no_sound"), ("actions", "no_click")) if cap not in self._caps)

    def _deliver(self, title, body, sound, token):
        bus = self._connect()
        Gio, GLib = self._gi
        server_sound = "sound" in self._caps
        hints = {"desktop-entry": GLib.Variant("s", "ouroboros")}
        if not sound:
            hints["suppress-sound"] = GLib.Variant("b", True)
        elif server_sound:
            hints["sound-name"] = GLib.Variant("s", _FD_SOUND)  # a themed name: the server's best effort
        actions = ["default", "Open"] if "actions" in self._caps else []
        text = GLib.markup_escape_text(body) if body and "body-markup" in self._caps else body
        try:
            reply = bus.call_sync(_FD_NAME, _FD_PATH, _FD_NAME, "Notify",
                                  GLib.Variant("(susssasa{sv}i)", ("Ouroboros", 0, "", title, text, actions, hints, -1)),
                                  GLib.VariantType.new("(u)"), Gio.DBusCallFlags.NONE, 3000, None)
        except Exception as exc:
            if _no_reply(Gio, exc):
                # Its id never came back, so a click on it cannot be told from another app's.
                return unconfirmed(self.platform, "no_reply")
            raise
        if actions:
            with self._lock:
                self._ids[int(reply.unpack()[0])] = token
                while len(self._ids) > _FD_LIMIT:
                    self._ids.pop(next(iter(self._ids)))
        if not sound:
            return submitted(self.platform, "off")
        return submitted(self.platform, "os" if server_sound else "none")


def native_notifier(on_click):
    """This platform's adapter, or None (the bridge then answers ``unavailable``)."""
    if sys.platform == "darwin":
        return MacNotifier(on_click)
    if sys.platform == "win32":
        return WindowsNotifier(on_click)
    if sys.platform.startswith("linux"):
        return FreedesktopNotifier(on_click)
    return None


def platform_name() -> str:
    return {"darwin": "macos", "win32": "windows"}.get(sys.platform, "linux" if sys.platform.startswith("linux")
                                                       else sys.platform)

"""Native fullscreen acknowledgement before shell replacement, without a live GUI.

A controlled AppKit event queue exercises the real lifecycle owner. Native macOS
qualification is separate: these tests cannot prove WindowServer Space retirement.
"""
from __future__ import annotations

from queue import Queue
import sys
import threading
from types import SimpleNamespace

import pytest

from ouroboros import config
from ouroboros.launcher_macos_handoff import MacOSWindowHandoff
import ouroboros.launcher_macos_handoff as handoff


class Event:
    def __init__(self):
        self.handlers = []

    def __iadd__(self, handler):
        self.handlers.append(handler)
        return self

    def emit(self):
        for handler in self.handlers:
            handler()


class Center:
    def __init__(self):
        self.observers = {}
        self.removed = []

    def addObserverForName_object_queue_usingBlock_(self, name, native, queue, callback):
        assert queue is None
        self.observers[name] = (native, callback)
        return name

    def removeObserver_(self, token):
        self.removed.append(token)
        self.observers.pop(token)

    def send(self, phase):
        self.observers[f"NSWindow{phase}FullScreenNotification"][1](None)


class Native:
    def __init__(self, center):
        self.center, self.mask, self.calls = center, 0, []
        self.fail_toggle = False
        self.behavior = 128 | 1
        self.delegate_value = object()
        self.visible = True

    def isVisible(self):
        return self.visible

    def orderOut_(self, _):
        self.visible = False

    def delegate(self):
        return self.delegate_value

    def setDelegate_(self, delegate):
        self.delegate_value = delegate

    def collectionBehavior(self):
        return self.behavior

    def setCollectionBehavior_(self, behavior):
        assert threading.current_thread() is threading.main_thread()
        self.behavior = behavior

    def styleMask(self):
        assert threading.current_thread() is threading.main_thread()
        return self.mask

    def toggleFullScreen_(self, _):
        assert threading.current_thread() is threading.main_thread()
        self.calls.append("exit")
        if self.fail_toggle:
            raise RuntimeError("native transition refused")
        self.center.send("WillExit")
        self.mask = 0  # AppKit may clear this before DidExit. It is NOT completion.


@pytest.fixture
def window(monkeypatch):
    center, queue = Center(), Queue()
    native = Native(center)
    window = SimpleNamespace(native=native, events=SimpleNamespace(before_show=Event(), closed=Event()))
    monkeypatch.setattr(handoff, "IS_MACOS", True)
    monkeypatch.setattr(config, "MACOS_HANDOFF_FULLSCREEN_WAIT_SEC", 1.0)
    monkeypatch.setattr(handoff, "_failure_delegate_type", lambda: SimpleNamespace(
        alloc=lambda: SimpleNamespace(init=lambda: SimpleNamespace())))
    monkeypatch.setitem(sys.modules, "AppKit", SimpleNamespace(
        NSWindowStyleMaskFullScreen=1, NSNotificationCenter=SimpleNamespace(defaultCenter=lambda: center),
        NSWindowCollectionBehaviorFullScreenNone=512, NSWindowCollectionBehaviorFullScreenPrimary=128,
        NSWindowCollectionBehaviorFullScreenAuxiliary=256))
    monkeypatch.setitem(sys.modules, "PyObjCTools", SimpleNamespace(
        AppHelper=SimpleNamespace(callAfter=lambda fn, *args: queue.put(lambda: fn(*args)))))
    owner = MacOSWindowHandoff()
    owner.attach(window)
    window.events.before_show.emit()
    return SimpleNamespace(owner=owner, native=native, center=center, queue=queue, window=window)


def start(window, cancelled=lambda: False):
    result = []
    thread = threading.Thread(target=lambda: result.append(window.owner.prepare(cancelled)))
    thread.start()
    action = window.queue.get(timeout=2)
    return thread, result, action


def join(thread, result, expected):
    thread.join(2)
    assert not thread.is_alive()
    assert result == [expected]


def test_windowed_handoff_is_ready_without_toggling(window):
    thread, result, action = start(window)
    action()
    join(thread, result, True)
    assert window.native.calls == []


def test_fullscreen_requires_did_exit_not_the_cleared_mask(window):
    window.native.mask = 1
    window.center.send("WillEnter")
    window.center.send("DidEnter")
    thread, result, action = start(window)
    action()
    assert window.native.calls == ["exit"] and window.native.mask == 0
    assert result == []
    window.owner._advance()  # another observation during the exit still cannot certify it
    assert result == []
    window.center.send("DidExit")
    join(thread, result, True)


def test_handoff_during_entry_defers_its_one_exit_until_did_enter(window):
    window.native.mask = 1
    window.center.send("WillEnter")
    thread, result, action = start(window)
    action()
    assert window.native.calls == []
    window.center.send("DidEnter")
    assert window.native.calls == ["exit"]
    window.center.send("DidExit")
    join(thread, result, True)


def test_handoff_during_an_owners_exit_does_not_toggle_back_in(window):
    window.center.send("WillExit")
    thread, result, action = start(window)
    action()
    assert window.native.calls == [] and result == []
    window.center.send("DidExit")
    join(thread, result, True)


@pytest.mark.parametrize("reason", ["cancel", "timeout"])
def test_abandoned_queued_action_never_toggles_later(window, monkeypatch, reason):
    window.native.mask = 1
    stop = threading.Event()
    if reason == "timeout":
        monkeypatch.setattr(config, "MACOS_HANDOFF_FULLSCREEN_WAIT_SEC", 0.01)
    thread, result, action = start(window, stop.is_set)
    if reason == "cancel":
        stop.set()
    join(thread, result, False)
    action()
    assert window.native.calls == []
    # Cancellation belongs to that request, not to every future source handoff.
    window.native.mask = 0
    thread, result, action = start(window)
    action()
    join(thread, result, True)


def test_cancellation_during_exit_keeps_shell_without_reversing_native_action(window):
    window.native.mask = 1
    stop = threading.Event()
    thread, result, action = start(window, stop.is_set)
    action()
    stop.set()
    join(thread, result, False)
    window.center.send("DidExit")
    assert window.native.calls == ["exit"]


def test_failed_exit_or_missing_ack_refuses_without_closing_window(window, monkeypatch):
    window.native.mask = 1
    window.native.fail_toggle = True
    thread, result, action = start(window)
    action()
    join(thread, result, False)
    window.native.fail_toggle = False
    monkeypatch.setattr(config, "MACOS_HANDOFF_FULLSCREEN_WAIT_SEC", 0.02)
    thread, result, action = start(window)
    action()
    join(thread, result, False)
    assert window.native.calls == ["exit", "exit"]


def test_unobserved_native_window_and_ui_thread_never_claim_ready(window):
    window.owner._phase = "unbound"
    thread, result, action = start(window)
    action()
    join(thread, result, False)
    assert window.owner.prepare(lambda: False) is False
    assert window.native.calls == []


def test_close_releases_observers_and_wakes_pending_worker(window):
    window.center.send("WillEnter")
    thread, result, action = start(window)
    action()
    window.window.events.closed.emit()
    join(thread, result, False)
    window.queue.get(timeout=2)()
    assert len(window.center.removed) == 4 and not window.center.observers


def test_cold_headless_and_other_platforms_import_no_appkit(monkeypatch):
    monkeypatch.setattr(handoff, "IS_MACOS", False)
    owner = MacOSWindowHandoff()
    owner.attach(object())
    assert owner.prepare(lambda: False) is True
    assert owner.prepare(lambda: True) is False
    monkeypatch.setattr(handoff, "IS_MACOS", True)
    assert MacOSWindowHandoff().prepare(lambda: False) is True


def test_manual_exit_before_queued_preparation_never_enters_again(window):
    window.native.mask = 1
    thread, result, action = start(window)
    window.center.send("WillExit")
    window.native.mask = 0
    window.center.send("DidExit")
    action()
    join(thread, result, True)
    assert window.native.calls == []
    assert window.native.behavior == 513  # unrelated collection bits survive


def test_reentry_is_disabled_during_cleanup_and_restored_after_failed_exec(window):
    observed = []
    thread = threading.Thread(target=lambda: observed.append(window.owner.before_release(
        lambda: False, lambda: observed.append(window.native.behavior))))
    thread.start()
    window.queue.get(timeout=2)()
    thread.join(2)
    assert not thread.is_alive() and observed == [513, True]
    assert window.native.behavior == 513
    window.owner.release()  # the launcher's existing on_failure path
    window.queue.get(timeout=2)()
    assert window.native.behavior == 129
    thread, result, action = start(window)
    action()
    join(thread, result, True)


def test_cancel_after_preparation_restores_controls_without_allowing_exec(window):
    stop, results, cleanup_ran = threading.Event(), [], threading.Event()

    def cleanup():
        cleanup_ran.set()
        stop.set()

    thread = threading.Thread(target=lambda: results.append(window.owner.before_release(stop.is_set, cleanup)))
    thread.start()
    window.queue.get(timeout=2)()
    join(thread, results, False)
    assert cleanup_ran.is_set()
    window.queue.get(timeout=2)()
    assert window.native.behavior == 129


@pytest.mark.parametrize("phase,mask", [("windowed", 0), ("fullscreen", 1)])
def test_explicit_native_failure_refuses_this_attempt_but_allows_a_future_one(window, phase, mask):
    window.center.send("WillEnter" if phase == "windowed" else "WillExit")
    thread, result, action = start(window)
    action()
    window.native.mask = mask
    window.owner._failed(phase)
    join(thread, result, False)
    thread, result, action = start(window)
    action()
    if mask:
        assert window.native.calls == ["exit"]
        window.center.send("DidExit")
    join(thread, result, True)


def test_delayed_release_cannot_undo_a_new_attempts_barrier(window):
    thread, result, action = start(window)
    action()
    join(thread, result, True)
    window.owner.release()
    window.owner.release()
    first, stale = window.queue.get(timeout=2), window.queue.get(timeout=2)
    first()
    thread, result, action = start(window)
    action()
    join(thread, result, True)
    stale()
    assert window.native.behavior == 513


def test_native_exit_does_not_leave_a_hidden_window_shown(window):
    window.native.mask = 1
    window.native.visible = False
    thread, result, action = start(window)
    action()
    window.native.visible = True  # AppKit's observed exit behavior
    window.center.send("DidExit")
    join(thread, result, True)
    assert window.native.visible is False

"""Cold Electron negotiation through real read/action logic; no native API or desktop use."""
from __future__ import annotations

import pytest

from skills.unix_computer_use import macos_native as native
from tests.test_unix_computer_use_element_act import FakeAX, deck
from tests.test_unix_computer_use_one_process import Clock
from tests.test_unix_computer_use_targeting import mac, receipt


class ColdAX(FakeAX):
    request_manual_accessibility = native._AX.request_manual_accessibility
    kCFBooleanTrue = object()

    def __init__(self, *, code=0, cold=True, ready=True, clock=None, trusted=True):
        super().__init__(deck(), windows=["win"], focused="win", trusted=trusted)
        self.code, self.cold, self.ready, self.clock = code, cold, ready, clock
        self.requests = []

    def cfs(self, name):
        return name

    def AXUIElementSetAttributeValue(self, app, name, value):
        assert self.timeouts, "set the messaging timeout before negotiating"
        assert (app, name, value) == ("app", "AXManualAccessibility", self.kCFBooleanTrue)
        self.requests.append((app, name, value))
        if self.code == 0 and self.ready:
            self.cold = False
        if self.clock:
            self.clock.t += 4  # consumes the action's existing budget, not a fresh one
        return self.code

    def attr(self, el, name):
        if el == "app":
            assert self.requests, "negotiate before any window/focus read"
            if self.cold:
                return None, -25212
        return super().attr(el, name)

    def window_refs(self, app):
        assert self.requests, "negotiate before enumerating windows"
        return ([], 0) if self.cold else super().window_refs(app)


def invoke(ax, operation, monkeypatch, *, emit=None, now=None):
    monkeypatch.setattr(native, "op_resolve", lambda q: {"ok": True, "candidates": [{"pid": 200, "app": "Cold"}]})
    def front():
        return {"pid": 100}
    if operation == "tree":
        return native.op_tree(200, ax=ax, window=39)
    if operation == "observe":
        return native.op_observe({"app": "pid:200", "window": "39"}, ax=ax, front=front)
    selector = {"identifier": "toolbar.play", "window": "39"}
    kwargs = {"now": now} if now else {}
    if operation == "act":
        return native.op_act(200, "press", selector, ax=ax, front=front, settle=lambda _: None, **kwargs)
    return native.op_element({"app": "pid:200", "do": "press", "select": selector, "observe": True},
                             ax=ax, front=front, settle=lambda _: None, emit=emit, **kwargs)


@pytest.mark.parametrize("operation", ["tree", "observe", "element"])
def test_cold_first_call_exposes_before_selection_and_preserves_action_observation(operation, monkeypatch):
    ax = ColdAX()
    checkpoints = []
    out = invoke(ax, operation, monkeypatch, emit=checkpoints.append)
    assert out["ok"] and out["ax_exposure"]["status"] == "accepted"
    assert len(ax.requests) == 1 and ax.balanced()
    if operation in ("act", "element"):
        assert ax.performed == [("play", "AXPress")]
        assert out["front_before"] == out["front_after"] == {"pid": 100}
    if operation == "element":
        assert out["observation"]["ok"] and out["observation"]["nodes"]
        assert len(checkpoints) == 1 and checkpoints[0]["ax_exposure"] == out["ax_exposure"]
        assert "observation" not in checkpoints[0]
    if operation in ("tree", "observe"):
        assert out["nodes"] and out["marks"] and out["window_id"] == 39


@pytest.mark.parametrize("code,status", [(-25205, "unsupported"), (-25208, "unsupported"),
                                       (-25204, "unknown")])
def test_negotiation_errors_remain_visible_without_breaking_an_available_tree(code, status, monkeypatch):
    ax = ColdAX(code=code, cold=False)
    out = invoke(ax, "element", monkeypatch)
    assert out["ok"] and out["ax_exposure"]["code"] == code and out["ax_exposure"]["status"] == status
    assert len(ax.requests) == len(ax.performed) == 1 and ax.balanced()


@pytest.mark.parametrize(("operation", "code"), [("tree", 0), ("tree", -25205), ("element", -25204)])
def test_accepted_is_not_readiness_and_errors_do_not_trigger_retries(code, operation, monkeypatch):
    ax = ColdAX(code=code, ready=False)
    out = invoke(ax, operation, monkeypatch)
    assert not out["ok"] and out["ax_exposure"]["code"] == code
    assert len(ax.requests) == 1 and not ax.performed and ax.balanced()


@pytest.mark.parametrize("operation", ["tree", "element"])
def test_untrusted_never_attempts_exposure(operation, monkeypatch):
    ax = ColdAX(trusted=False)
    out = invoke(ax, operation, monkeypatch)
    assert out["reason"] == "accessibility_untrusted"
    assert not ax.requests and not ax.performed and ax.balanced()


def test_exposure_time_spends_the_action_budget(monkeypatch):
    clock = Clock()
    ax = ColdAX(clock=clock)
    out = invoke(ax, "element", monkeypatch, now=clock)
    assert out["reason"] == "budget_exhausted" and out["dispatched"] is False
    assert not ax.performed and ax.balanced()


def test_later_calls_renegotiate_without_a_pid_cache(monkeypatch):
    ax = ColdAX()
    assert invoke(ax, "tree", monkeypatch)["ok"]
    ax.cold = True  # renderer rebuilt (or a PID reused); earlier success says nothing about this call
    assert invoke(ax, "element", monkeypatch)["ok"]
    assert len(ax.requests) == 2 and len(ax.performed) == 1 and ax.balanced()


@pytest.mark.parametrize("ready", [True, False])
def test_public_tree_keeps_exposure_receipt_even_when_the_window_is_not_ready(ready, tmp_path, monkeypatch):
    ax = ColdAX(ready=ready)
    monkeypatch.setattr(native, "op_resolve", lambda q: {"ok": True, "candidates": [{"pid": 200, "app": "Cold"}]})

    def helper(args, _state):
        assert args[0] == "observe"
        return native.op_observe(args[1], ax=ax)

    _module, api, state = mac(tmp_path, monkeypatch, helper=helper)
    out = receipt(api.tools["ax_tree"](app="pid:200", window="39"))
    assert out["ok"] is ready and out["ax_exposure"]["status"] == "accepted"
    assert len(ax.requests) == len(state["helper"]) == 1 and ax.balanced()

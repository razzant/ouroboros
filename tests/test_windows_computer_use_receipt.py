"""``element_act`` receipts and the host's effect mark (windows_computer_use 0.1.1).

``observed_effect: true`` is the PluginAPI literal for an executed external effect
(``ouroboros/contracts/plugin_api.py``); the result reviewer keeps such receipts
verbatim (``ouroboros/tools/review_result.py``). Here it means that a mutating pattern
call returned on the window named in ``delivered_to``. Each case is two-sided: the
path that earns the mark for a completed mutating call carries none for a view-only
act, an action that raised, a worker that did not answer, or a refusal. No Windows,
COM or desktop is involved: the act path runs over the comtypes-faithful fakes of
``test_windows_computer_use_uia`` and answers as the host reads it, as JSON text.
"""

from __future__ import annotations

import json
import sys
import types

import pytest

from tests._windows_computer_use import load_package
from tests.test_windows_computer_use_uia import Automation, Calls, COMError, dialog, make, window
from tests.test_windows_computer_use_uia import fake_comtypes  # noqa: F401 (shared autouse fixture)

load_package()
from windows_computer_use import core, plugin, tree, win_uia  # noqa: E402

MUTATING = ("invoke", "set_value", "toggle", "expand", "collapse", "select")
VIEW_ONLY = ("focus", "scroll_into_view")
WINDOW = {"hwnd": "0x1", "title": "dialog", "pid": 42}  # the shared fake window(), pid as the guard reads it


def plan(act):
    return core.act_plan({"hwnd": 1, "do": act, "select": {"automation_id": "go"},
                          **({"text": ""} if act == "set_value" else {})})


def test_every_verb_is_either_mutating_or_view_only():
    assert sorted(MUTATING + VIEW_ONLY) == sorted(core.ACTS)
    assert sorted(act for act in core.ACTS if not core.act_mutating(act)) == sorted(VIEW_ONLY)


@pytest.mark.parametrize("act", MUTATING)
def test_a_completed_mutating_receipt_carries_the_mark_and_its_window(act):
    said = core.receipt(plan(act), {"role": "button"}, {"role": "button"}, 5, 1, delivered_to=WINDOW)
    assert said["observed_effect"] is True and said["delivered_to"] == WINDOW
    assert said["ok"] is True and said["mutating"] is True and said["action_outcome"] == "completed"
    bare = core.receipt(plan(act), {}, {}, 5, 1)  # the 0.1.0 positional call still works
    assert bare["observed_effect"] is True and "delivered_to" not in bare


@pytest.mark.parametrize("act", VIEW_ONLY)
def test_a_view_only_receipt_names_its_window_without_the_mark(act):
    said = core.receipt(plan(act), {"role": "button"}, {"role": "button"}, 5, 1, delivered_to=WINDOW)
    assert "observed_effect" not in said and said["delivered_to"] == WINDOW
    assert said["ok"] is True and said["mutating"] is False and said["action_outcome"] == "completed"


@pytest.fixture
def element_act(monkeypatch):
    """The registered handler as the host calls it, with only the native edges faked:
    the desktop check, window resolution, the comtypes wrapper and the COM provider."""
    current = {}
    fake_win32 = types.ModuleType("windows_computer_use.win32fb")
    fake_win32.resolve_window = lambda hwnd: current["window"]
    fake_win32.window_pid = lambda hwnd: 42
    monkeypatch.setitem(sys.modules, "windows_computer_use.win32fb", fake_win32)
    monkeypatch.setattr(sys.modules["windows_computer_use"], "win32fb", fake_win32, raising=False)
    monkeypatch.setattr(plugin, "_ensure_windows", lambda: None)
    monkeypatch.setattr(win_uia, "client_module", lambda: None)
    monkeypatch.setattr(win_uia, "thread_automation", lambda: Automation(current["root"]))
    handler = plugin._json_tool(plugin.WindowsComputerUse(None).element_act)

    def run(root, do, select, on=None):
        current.update(root=root, window=on or window())
        return json.loads(handler(hwnd=1, select=select, do=do, timeout_ms=0))

    return run


def test_the_host_reads_the_mark_on_a_completed_invoke_and_not_on_focus(element_act):
    calls = Calls()
    root = dialog(calls, make(calls, "go", 50000, name="Go", patterns={10000: None}))
    pressed = element_act(root, "invoke", {"automation_id": "go"})
    focused = element_act(root, "focus", {"automation_id": "go"})
    assert calls == [("Invoke", "go"), ("SetFocus", "go")]
    assert pressed["observed_effect"] is True and pressed["delivered_to"] == WINDOW
    assert "observed_effect" not in focused and focused["delivered_to"] == WINDOW


def test_an_unread_title_is_named_in_the_delivery_target(element_act):
    calls = Calls()
    root = dialog(calls, make(calls, "go", 50000, patterns={10000: None}))
    untitled = tree.WindowInfo(1, "", "Dialog", 42, None, True, "argument", 96, ("title",))
    said = element_act(root, "invoke", {"automation_id": "go"}, on=untitled)
    assert said["observed_effect"] is True
    assert said["delivered_to"] == {"hwnd": "0x1", "title": "", "pid": 42, "unread": ["title"]}


def test_an_action_that_raised_has_an_unknown_outcome_and_no_mark(element_act):
    calls = Calls()
    target = make(calls, "go", 50000, patterns={10000: None})

    def raised():
        calls.append(("Invoke", "go"))
        raise COMError(-2147220991, "element not available")

    target.patterns[10000].Invoke = raised
    said = element_act(dialog(calls, target), "invoke", {"automation_id": "go"})
    assert calls == [("Invoke", "go")]
    assert said["ok"] is False and said["action_call_made"] is True and said["action_outcome"] == "unknown"
    assert "observed_effect" not in said and said["delivered_to"] == WINDOW


def test_a_worker_that_did_not_answer_carries_no_mark(element_act, monkeypatch):
    calls = Calls()
    monkeypatch.setattr(plugin._Worker, "run", lambda self: None)  # abandoned: no answer at all
    said = element_act(dialog(calls, make(calls, "go", 50000, patterns={10000: None})),
                       "invoke", {"automation_id": "go"})
    assert said["ok"] is False and said["action_call_made"] == "unknown" and said["action_outcome"] == "unknown"
    assert "observed_effect" not in said and "delivered_to" not in said


def test_refusals_carry_no_mark(element_act):
    calls = Calls()
    twins = dialog(calls, make(calls, "a", 50000, name="Go", patterns={10000: None}),
                   make(calls, "b", 50000, name="Go", patterns={10000: None}))
    ambiguous = element_act(twins, "invoke", {"name": "Go"})
    unsupported = element_act(dialog(calls, make(calls, "label", 50020, name="Status")), "invoke", {"name": "Status"})
    assert ambiguous["reason"] == "ambiguous" and "does not support InvokePattern" in unsupported["error"]
    for refused in (ambiguous, unsupported):
        assert refused["ok"] is False and "observed_effect" not in refused and "delivered_to" not in refused
    assert calls == []
